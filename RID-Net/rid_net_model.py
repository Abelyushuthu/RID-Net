"""
RID-Net: Rule-Injected Disentangled Network

Dual-tower:
  3D Tower: Simple3DGNN -> emb_3D [64]
  1D Tower: MACCS [166] -> MLP -> emb_1D [64]
  Predict: concat -> MLP -> logS
  SSR-Loss: d(pred)/d(MACCS) constrained by chemical rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tqdm import tqdm

ATOMIC_NUMBERS = [6, 7, 8, 9, 15, 16, 17, 5, 35, 53, 34]
NUM_NODE_TYPES = len(ATOMIC_NUMBERS) + 1
ELE_TO_NODETYPE = {e: i for i, e in enumerate(ATOMIC_NUMBERS)}

DATASET_CONFIGS = {
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "target_col": "measured log solubility in mols per litre",
        "rules": [
            (15,1,1.0),(33,1,1.0),(89,1,1.0),(62,1,0.8),(115,1,0.7),(55,1,0.6),(41,1,0.5),
            (144,-1,1.0),(126,-1,0.8),(155,-1,0.7),(156,-1,0.7),(154,-1,0.7),(106,-1,0.6),
        ],
    },
    "lipo": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "target_col": "exp",
        "rules": [
            (15,-1,1.0),(33,-1,1.0),(89,-1,1.0),(62,-1,0.8),(115,-1,0.7),(55,-1,0.6),(41,-1,0.5),
            (144,1,1.0),(126,1,0.8),(155,1,0.7),(156,1,0.7),(106,1,0.6),
        ],
    },
    "freesolv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "target_col": "expt",
        "rules": [
            (15,-1,1.0),(33,-1,1.0),(89,-1,1.0),(62,-1,0.8),
            (144,1,1.0),(126,1,0.8),(155,1,0.7),(106,1,0.6),
        ],
    },
}


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_g=32):
        super().__init__()
        offset = torch.linspace(start, stop, num_g)
        widths = (offset[1]-offset[0]) * torch.ones_like(offset)
        self.register_buffer("offset", offset)
        self.register_buffer("widths", widths)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        o = self.offset.unsqueeze(0).unsqueeze(0)
        w = self.widths.unsqueeze(0).unsqueeze(0)
        return torch.exp(-0.5 * ((dist - o) / w) ** 2)


class Simple3DGNN(nn.Module):
    def __init__(self, out_dim=64, h_dim=64, num_g=32, cutoff=5.0):
        super().__init__()
        self.atom_embed = nn.Embedding(NUM_NODE_TYPES, h_dim)
        self.dist_exp = GaussianSmearing(0.0, cutoff, num_g)
        self.edge_net = nn.Sequential(
            nn.Linear(num_g, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU())
        self.node_update = nn.Sequential(
            nn.Linear(h_dim*2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU())
        self.readout = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, out_dim))

    def forward(self, atomic_numbers_list, pos_list):
        device = next(self.parameters()).device
        embs = []
        for elements, pos in zip(atomic_numbers_list, pos_list):
            elements = elements.to(device)
            pos = pos.to(device)
            h = self.atom_embed(elements)
            dist = torch.cdist(pos, pos).clamp(min=1e-6)
            edge_feat = self.dist_exp(dist)
            edge_msg = self.edge_net(edge_feat)
            neighbor_h = h.unsqueeze(0).expand(h.size(0), -1, -1)
            messages = edge_msg * neighbor_h
            agg_msg = messages.sum(dim=1)
            h_new = self.node_update(torch.cat([h, agg_msg], dim=-1))
            embs.append(self.readout(h_new).mean(dim=0))
        return torch.stack(embs, dim=0)


class OneDTower(nn.Module):
    def __init__(self, fp_dim=166, h_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, h_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class RIDNet(nn.Module):
    def __init__(self, emb_dim=64, h_dim=128):
        super().__init__()
        self.encoder_3d = Simple3DGNN(out_dim=emb_dim)
        self.tower_1d = OneDTower(out_dim=emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim*2, h_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, 1))

    def forward(self, atomic_numbers_list, pos_list, fp_list):
        emb_3d = self.encoder_3d(atomic_numbers_list, pos_list)
        device = emb_3d.device
        fp = torch.stack(fp_list, 0).to(device).float()
        fp.requires_grad_(True)
        emb_1d = self.tower_1d(fp)
        pred = self.predictor(torch.cat([emb_3d, emb_1d], dim=-1))
        return pred, emb_3d, emb_1d, fp




class RIDNet_1DOnly(nn.Module):
    """Ablation: remove 3D tower. Only MACCS -> MLP -> prediction."""
    def __init__(self, emb_dim=64, h_dim=128):
        super().__init__()
        self.tower_1d = OneDTower(out_dim=emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, h_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h_dim, 1))

    def forward(self, atomic_numbers_list, pos_list, fp_list):
        device = next(self.parameters()).device
        fp = torch.stack(fp_list, 0).to(device).float()
        fp.requires_grad_(True)
        emb_1d = self.tower_1d(fp)
        pred = self.predictor(emb_1d)
        return pred, None, emb_1d, fp


def ssr_loss(pred, raw_fp, rules, lam=0.3):
    B = pred.size(0)
    loss = torch.tensor(0.0, device=pred.device)
    for bit, direction, weight in rules:
        gm = torch.autograd.grad(
            pred.mean(), raw_fp, create_graph=True, retain_graph=True)[0]
        g = B * gm[:, bit]
        loss = loss + weight * F.relu(-direction * g + 0.001).mean()
    return lam * loss


def gen_conf(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(mol, p) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=500)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return Chem.RemoveHs(mol)


def preprocess(df, target_col):
    els_list, pos_list, fp_list, labels = [], [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            mol = gen_conf(row["smiles"], seed=i)
            conf = mol.GetConformer()
            els = torch.tensor(
                [ELE_TO_NODETYPE.get(a.GetAtomicNum(), NUM_NODE_TYPES-1)
                 for a in mol.GetAtoms()], dtype=torch.long)
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
            pos = pos - pos.mean(dim=0, keepdim=True)
            fp = torch.tensor(
                np.array(MACCSkeys.GenMACCSKeys(mol)),
                dtype=torch.float)[1:]
            els_list.append(els)
            pos_list.append(pos)
            fp_list.append(fp)
            labels.append(row[target_col])
        except:
            pass
    return els_list, pos_list, fp_list, torch.tensor(labels, dtype=torch.float)


def scaffold_split(df):
    scaf = defaultdict(list)
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol:
            scaf[MurckoScaffold.MurckoScaffoldSmiles(mol=mol)].append(i)
    train, val, test = [], [], []
    for sset in sorted(scaf.values(), key=len, reverse=True):
        if len(test) < int(len(df)*0.2):
            test.extend(sset)
        elif len(val) < int(len(df)*0.1):
            val.extend(sset)
        else:
            train.extend(sset)
    return train, val, test
