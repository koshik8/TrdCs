import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optz
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tfs
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Reproducibility ────────────────────────────────────────────────────────
RAND_SEED = 42
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RAND_SEED)
# ──────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
# SparseLinearLayer
# ══════════════════════════════════════════════════════════════════

class SparseLinearLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int,
                 gate_thresh: float = 0.5):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.gate_thresh = gate_thresh

        self.w_param = nn.Parameter(torch.empty(out_dim, in_dim))
        self.b_param = nn.Parameter(torch.zeros(out_dim))

        self.gate_param = nn.Parameter(
            torch.full((out_dim, in_dim), 3.0)
        )

        nn.init.kaiming_uniform_(self.w_param, a=0.01)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        gate_vals = torch.sigmoid(self.gate_param)

        if self.training:
            eff_w = self.w_param * gate_vals
        else:
            hard_mask = (gate_vals > self.gate_thresh).float()
            eff_w = self.w_param * hard_mask

        return F.linear(inp, eff_w, self.b_param)

    def soft_gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_param).detach()

    def hard_gates(self) -> torch.Tensor:
        return (torch.sigmoid(self.gate_param) > self.gate_thresh).float().detach()

    def sparsity_ratio(self) -> float:
        return (self.hard_gates() == 0).float().mean().item()

    def extra_repr(self):
        return (f"in_dim={self.in_dim}, "
                f"out_dim={self.out_dim}, "
                f"gate_thresh={self.gate_thresh}")


# ══════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════

class SparseCNN(nn.Module):
    
    def __init__(self, gate_thresh: float = 0.5):
        super().__init__()

        self.feature_stack = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )

        self.lin1  = SparseLinearLayer(2048, 512, gate_thresh=gate_thresh)
        self.bn_l1 = nn.BatchNorm1d(512)
        self.drop_l = nn.Dropout(p=0.4)
        self.lin2  = SparseLinearLayer(512, 10, gate_thresh=gate_thresh)

    def forward(self, inp):
        inp = self.feature_stack(inp)
        inp = inp.view(inp.size(0), -1)
        inp = self.drop_l(F.relu(self.bn_l1(self.lin1(inp))))
        return self.lin2(inp)

    def sparse_layers(self):
        for mod in self.modules():
            if isinstance(mod, SparseLinearLayer):
                yield mod

    def sparsity_penalty(self) -> torch.Tensor:
        
        gates_all = [
            torch.sigmoid(layer.gate_param)
            for layer in self.sparse_layers()
        ]
        return torch.cat([g.reshape(-1) for g in gates_all]).mean()

    def total_sparsity(self) -> float:
        tot, pruned = 0, 0
        for layer in self.sparse_layers():
            mask = layer.hard_gates()
            pruned += (mask == 0).sum().item()
            tot  += mask.numel()
        return pruned / tot if tot > 0 else 0.0

    def gate_values(self) -> np.ndarray:
        return np.concatenate([
            layer.soft_gates().cpu().numpy().ravel()
            for layer in self.sparse_layers()
        ])

    def weight_params(self):
        for n, p in self.named_parameters():
            if "gate_param" not in n:
                yield p

    def gate_params(self):
        for n, p in self.named_parameters():
            if "gate_param" in n:
                yield p


# ══════════════════════════════════════════════════════════════════
# Compression
# ══════════════════════════════════════════════════════════════════

def build_compressed(model: SparseCNN) -> nn.Module:
    
    model.eval()
    layers = list(model.sparse_layers())
    new_layers  = []
    prev_idx   = None

    for i, layer in enumerate(layers):
        with torch.no_grad():
            mask     = layer.hard_gates()
            w      = layer.w_param.data.clone() * mask
            b      = layer.b_param.data.clone()

            if prev_idx is not None:
                w = w[:, prev_idx]

            last   = (i == len(layers) - 1)
            if last:
                active = torch.ones(w.size(0), dtype=torch.bool)
            else:
                active = (mask.sum(dim=1) > 0)

            kept = active.nonzero(as_tuple=True)[0]
            w = w[kept]
            b = b[kept]

            lin = nn.Linear(w.size(1), w.size(0))
            lin.weight.data = w
            lin.bias.data   = b
            new_layers.append(lin)

            pct = kept.numel() / mask.size(0) * 100
            print(f"  FC{i+1}: {mask.size()} → "
                  f"({kept.numel()} × {w.size(1)})  "
                  f"[{pct:.1f}% neurons retained]")
            prev_idx = kept

    seq = []
    for idx, fc in enumerate(new_layers):
        seq.append(fc)
        if idx < len(new_layers) - 1:
            seq.append(nn.ReLU(inplace=True))

    return nn.Sequential(*seq)


# ══════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════

def data_loaders(batch_sz=128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    tr = tfs.Compose([
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(32, padding=4),
        tfs.ColorJitter(brightness=0.2, contrast=0.2),
        tfs.ToTensor(),
        tfs.Normalize(mean, std),
    ])
    te = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean, std),
    ])

    gpu   = torch.cuda.is_available()
    workers = 2 if gpu else 0

    train_ds = torchvision.datasets.CIFAR10("./data", True,  download=True, transform=tr)
    test_ds  = torchvision.datasets.CIFAR10("./data", False, download=True, transform=te)

    train_dl = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                          num_workers=workers, pin_memory=gpu)
    test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=workers, pin_memory=gpu)
    return train_dl, test_dl


# ══════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════

def train_epoch(net, loader, opt, lam_sparse, device):
    net.train()
    total_loss = correct = n = 0

    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad()
        logits = net(imgs)

        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = net.sparsity_penalty()
        loss        = cls_loss + lam_sparse * sparse_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_model(net, loader, device):
    net.eval()
    correct = n = 0
    for imgs, labels in loader:
        imgs   = imgs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        correct += (net(imgs).argmax(1) == labels).sum().item()
        n       += imgs.size(0)
    return correct / n


def train_full(lam_sparse, epochs, device, train_dl, test_dl):
    print(f"\n{'='*58}")
    print(f"  Training  λ = {lam_sparse}  ({epochs} epochs)")
    print(f"{'='*58}")

    net = SparseCNN(gate_thresh=0.5).to(device)

    opt = optz.Adam([
        {"params": list(net.weight_params()), "lr": 1e-3},
        {"params": list(net.gate_params()),   "lr": 5e-3},
    ], weight_decay=1e-4)

    sched = optz.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            net, train_dl, opt, lam_sparse, device
        )
        sched.step()

        if ep % 2 == 0 or ep == epochs:
            net.eval()
            sparsity = net.total_sparsity()
            test_acc = eval_model(net, test_dl, device)
            net.train()

            mean_gate = net.gate_values().mean()
            print(
                f"  Ep {ep:>3}/{epochs}  "
                f"loss={train_loss:.3f}  "
                f"train={train_acc*100:.1f}%  "
                f"test={test_acc*100:.1f}%  "
                f"sparsity={sparsity*100:.1f}%  "
                f"mean_gate={mean_gate:.3f}"
            )

    final_acc      = eval_model(net, test_dl, device)
    final_sparsity = net.total_sparsity()
    mean_gate      = net.gate_values().mean()

    print(f"\n  ✓ Test Accuracy  : {final_acc*100:.2f}%")
    print(f"  ✓ Sparsity Level : {final_sparsity*100:.2f}%")
    print(f"  ✓ Mean Gate Value: {mean_gate:.4f}")
    return final_acc, final_sparsity, net


# ══════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════

def plot_gates(net, lam_val, save_path):
    gates = net.gate_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(gates, bins=100)
    axes[0].axvline(0.5)
    axes[0].set_title("Full distribution")

    low = gates[gates < 0.6]
    axes[1].hist(low, bins=80)
    axes[1].axvline(0.5)
    axes[1].set_title("Zoomed")

    plt.savefig(save_path)
    plt.close()


def plot_trade(res, save_path):
    lambdas    = [r["lambda"] for r in res]
    accs       = [r["accuracy"] * 100 for r in res]
    sparsities = [r["sparsity"] * 100 for r in res]

    fig, ax1 = plt.subplots()
    ax1.plot(lambdas, accs)

    ax2 = ax1.twinx()
    ax2.plot(lambdas, sparsities)

    plt.savefig(save_path)
    plt.close()


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def run():
   
    EPOCHS      = 40
    BATCH_SIZE  = 128
    LAMBDAS     = [0.1, 0.25, 0.55]
    BEST_LAMBDA = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs", exist_ok=True)
    train_dl, test_dl = data_loaders(BATCH_SIZE)

    results    = []
    best_model = None

    for lam in LAMBDAS:
        acc, sparsity, model = train_full(lam, EPOCHS, device, train_dl, test_dl)
        results.append({"lambda": lam, "accuracy": acc, "sparsity": sparsity})
        if lam == BEST_LAMBDA:
            best_model = model

    if best_model is not None:
        build_compressed(best_model)
        plot_gates(best_model, BEST_LAMBDA, "outputs/gate_distribution.png")

    plot_trade(results, "outputs/accuracy_sparsity_tradeoff.png")


if __name__ == "__main__":
    run()