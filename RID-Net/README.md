# RID-Net: Rule-Injected Disentangled Network

**BIBM 2026 Submission** - Molecular Property Prediction with Dual-Tower Architecture + Chemical Rule Injection.

## Architecture

  3D Tower: Simple3DGNN -> emb_3D [64]
  1D Tower: MACCS FP [166] -> MLP -> emb_1D [64]
  Predict: concat -> MLP -> logS
  SSR-Loss: d(pred)/d(MACCS) constrained by chemical rules

## Setup

  pip install -r requirements.txt

## Usage

  # Single dataset
  python train_esol.py --dataset esol

  # All datasets
  python train_esol.py --dataset all

  # Without SSR-Loss (ablation)
  python train_esol.py --dataset esol --no-ssr

## Datasets

  ESOL: Water solubility, 1,128 molecules, logS
  Lipophilicity: logP, 4,200 molecules
  FreeSolv: Solvation free energy, 643 molecules

## Files

  rid_net_model.py    Model: Simple3DGNN + OneDTower + SSR-Loss
  train_esol.py       Training pipeline (multi-dataset, multi-seed)
  requirements.txt    Dependencies: torch, rdkit, numpy, sklearn, pandas, tqdm
  README.md           This file
