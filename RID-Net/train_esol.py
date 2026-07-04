"""
train.py - Multi-dataset molecular property prediction with RID-Net
"""

import sys, os, json, random, copy, urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from rid_net_model import (
    RIDNet_1DOnly,
    RIDNet, ssr_loss, DATASET_CONFIGS, preprocess, scaffold_split,
)

class MolDataset(Dataset):
    def __init__(self, e, p, f, l):
        self.e, self.p, self.f, self.l = e, p, f, l
    def __len__(self):
        return len(self.l)
    def __getitem__(self, i):
        return self.e[i], self.p[i], self.f[i], self.l[i]

def collate(b):
    e, p, f, l = zip(*b)
    return list(e), list(p), list(f), torch.stack(l)

def train_one(model, loader, opt, device, rules):
    model.train()
    tl, n = 0.0, 0
    for e, p, f, l in tqdm(loader, desc='Train', leave=False):
        l = l.to(device)
        pred, _, _, raw_fp = model(e, p, f)
        lm = F.mse_loss(pred.squeeze(), l)
        ls = ssr_loss(pred, raw_fp, rules=rules) if rules else torch.tensor(0.0, device=device)
        loss = lm + ls
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item() * len(l)
        n += len(l)
    return tl / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ap, at = [], []
    for e, p, f, l in loader:
        l = l.to(device)
        pred, _, _, _ = model(e, p, f)
        ap.append(pred.view(-1).cpu())
        at.append(l.cpu())
    p = torch.cat(ap).numpy()
    t = torch.cat(at).numpy()
    return {
        "rmse": np.sqrt(mean_squared_error(t, p)),
        "mae": np.abs(p - t).mean(),
        "r2": r2_score(t, p),
    }

def run_experiment(df, cfg, train_idx, val_idx, test_idx, device, seed=42, epochs=200, use_ssr=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    all_e, all_p, all_f, all_y = preprocess(df, cfg["target_col"])

    tr = DataLoader(
        MolDataset([all_e[i] for i in train_idx], [all_p[i] for i in train_idx],
                   [all_f[i] for i in train_idx], all_y[train_idx]),
        batch_size=32, shuffle=True, collate_fn=collate)
    va = DataLoader(
        MolDataset([all_e[i] for i in val_idx], [all_p[i] for i in val_idx],
                   [all_f[i] for i in val_idx], all_y[val_idx]),
        batch_size=32, collate_fn=collate)
    te = DataLoader(
        MolDataset([all_e[i] for i in test_idx], [all_p[i] for i in test_idx],
                   [all_f[i] for i in test_idx], all_y[test_idx]),
        batch_size=32, collate_fn=collate)

    model = RIDNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)
    rules = cfg["rules"] if use_ssr else []
    bv, bt = float("inf"), None

    for ep in range(1, epochs + 1):
        train_one(model, tr, opt, device, rules)
        vm = evaluate(model, va, device)
        sched.step(vm["rmse"])
        if vm["rmse"] < bv:
            bv = vm["rmse"]
            bt = evaluate(model, te, device)
        if ep % 50 == 0:
            lr = opt.param_groups[0]["lr"]
            msg = "  E{:3d} | val={:.4f} | test={:.4f} | lr={:.2e}".format(ep, vm["rmse"], bt["rmse"], lr)
            print(msg)

    return bt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="esol",
                        choices=["esol", "lipo", "freesolv", "all"])
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--use_ssr", action="store_true", default=True)
    parser.add_argument("--no_ssr", action="store_false", dest="use_ssr")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_ssr", "1d_only", "1d_ssr", "all"],
                        help="Run ablation experiment instead of full pipeline")
    args = parser.parse_args()

    print("Device: {} | SSR: {} | Epochs: {}".format(args.device, args.use_ssr, args.epochs))
    datasets = [args.dataset] if args.dataset != "all" else ["esol", "lipo", "freesolv"]
    all_results = {}

    for ds in datasets:
        print()
        print("=" * 60)
        print("  DATASET: " + ds.upper())
        print("=" * 60)
        cfg = DATASET_CONFIGS[ds]
        df = pd.read_csv(cfg["url"])
        print("  Molecules: {}".format(len(df)))
        split_results = {"random": {}, "scaffold": {}}

        for st in ["random", "scaffold"]:
            for seed in range(42, 42 + args.n_seeds):
                print()
                print("  [{}] seed={} | SSR={}".format(st, seed, "ON" if args.use_ssr else "OFF"))
                if st == "random":
                    ti, te = train_test_split(range(len(df)), test_size=0.2, random_state=seed)
                    ti, va = train_test_split(ti, test_size=0.125, random_state=seed)
                else:
                    ti, va, te = scaffold_split(df)
                tm = run_experiment(df, cfg, ti, va, te, args.device,
                                    seed=seed, epochs=args.epochs, use_ssr=args.use_ssr)
                split_results[st]["s{}".format(seed)] = {k: float(v) for k, v in tm.items()}
                print("    TEST: RMSE={:.4f} MAE={:.4f} R2={:.4f}".format(tm["rmse"], tm["mae"], tm["r2"]))

        all_results[ds] = split_results
        for st in ["random", "scaffold"]:
            if not split_results[st]:
                continue
            rms = [v["rmse"] for v in split_results[st].values()]
            print("  {}: RMSE = {:.4f} +- {:.4f}".format(st, np.mean(rms), np.std(rms)))

    print()
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for ds, res in all_results.items():
        for st in ["random", "scaffold"]:
            if not res.get(st):
                continue
            rms = [v["rmse"] for v in res[st].values()]
            print("  {:10s} {:8s}: RMSE = {:.4f} +- {:.4f}".format(ds, st, np.mean(rms), np.std(rms)))

    os.makedirs("results", exist_ok=True)
    with open("results/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Saved to results/all_results.json")


if __name__ == "__main__":
    main()