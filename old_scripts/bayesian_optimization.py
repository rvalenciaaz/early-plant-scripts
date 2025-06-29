#!/usr/bin/env python3
"""
Bayesian Optimisation over Evo 2 embeddings (multi‑objective: expression ↑, binding ↑)
==============================================================================

This script demonstrates how to use Evo 2 sequence embeddings as a *continuous, learned
representation* of protein sequences and run **multi‑objective Bayesian optimisation**
in BoTorch to propose new nanobody variants that simultaneously maximise
(1) heterologous *expression* level and (2) *binding* affinity (to be measured
externally).

Pipeline overview
-----------------
1. **Seed data** — CSV with columns: `sequence` (AA letters), `expr` (float),
   `bind` (float or NaN if still unknown).
2. **Embed** — call NVIDIA/Arc “Evo 2” API to obtain a 4096‑d (or configurable)
   vector per sequence.
3. **Fit surrogate GP** — a *multi‑output* Gaussian Process (`MultiTaskGP`) on
   the embedding space → two objectives.
4. **Acquisition** — use *q‑EHVI* (or *HVKG* for decoupled costs) to propose
   batches of new embeddings that trade off both objectives.
5. **Decode / Generate sequence** — map embeddings back to *actual sequences*.
   Here we sample nearby mutants by gradient‑guided hot‑spotting; for production
   you may replace this with Evo 2 *generative* completions or constrained
   sequence search.
6. **External evaluation loop** — synthesise / express / measure new variants,
   append results to CSV, and re‑run optimisation.

This script focuses on steps 2–4 and provides stubs for decoding + evaluation.

Requirements
~~~~~~~~~~~~
- Python ≥3.10
- torch, gpytorch, botorch (>=0.14)
- pandas, numpy, tqdm, requests
- Optional: scikit‑learn (for a baseline Ridge regressor on embeddings → expr)

Usage (simplest loop)
~~~~~~~~~~~~~~~~~~~~~
```bash
python evo2_botorch_bo.py \
    --seed-data nanobody_initial.csv \
    --iterations 10 \
    --batch-size 8 \
    --expr-col expr --bind-col bind \
    --api-key $NVIDIA_API_KEY
```
After every BO iteration, a CSV `proposals_round_<n>.csv` is printed containing
fresh sequences to measure.

Notes
~~~~~
* **Decoupled objectives** — if expression can be predicted cheaply *in silico*
  (e.g. via a regression head we fit here) while binding requires wet‑lab data,
  switch `--decoupled` to use BoTorch’s HVKG acquisition.
* **Ethics & IP** — generated sequences may inherit training‑set biases; ensure
  downstream validation.
* **Decode step** — non‑trivial; current placeholder enumerates 1‑AA mutants of
  best upstream sequence.  Replace with a proper generator for high‑dim spaces.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.max_value_entropy_search import qMultiObjectiveEntropy
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from requests import Session
from tqdm import tqdm

# ---------------------------------------------------------
# Evo 2 Embedding client
# ---------------------------------------------------------

class Evo2Client:
    """Lightweight wrapper for the NVIDIA GenAI Evo 2 embedding endpoint."""

    _ENDPOINT = "https://api.nvidia.com/v1/genai/evo2/embeddings"

    def __init__(self, api_key: str, model: str = "evo2-40b-embed") -> None:
        self.session = Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self.model = model

    def embed(self, sequences: List[str]) -> np.ndarray:
        """Return an [N, D] array of embeddings (float32)."""
        payload = {"model": self.model, "sequences": sequences}
        r = self.session.post(self._ENDPOINT, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return np.asarray(data["embeddings"], dtype=np.float32)

# ---------------------------------------------------------
# Surrogate model utilities
# ---------------------------------------------------------

def _fit_gp(x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(x, y, outcome_transform=Standardize(m=y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.train()
    torch.optim.SGD(model.parameters(), lr=0.1).zero_grad()
    mll.backward()
    torch.optim.LBFGS(model.parameters()).step(lambda: -mll())
    model.eval()
    return model

# ---------------------------------------------------------
# Candidate generation helpers
# ---------------------------------------------------------

def propose_embeddings(model: SingleTaskGP, ref_point: torch.Tensor, batch_size: int, qmc: bool = True) -> torch.Tensor:
    """Optimise qEHVI in the embedding space and return *batch_size* points."""
    partitioning = NondominatedPartitioning(num_outcomes=2, Y=model.train_targets)
    sampler = SobolQMCNormalSampler(num_samples=256) if qmc else None

    acqf = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )

    bounds = torch.stack([
        model.train_inputs[0].min(0).values - 0.1,
        model.train_inputs[0].max(0).values + 0.1,
    ])

    new_x, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=256,
    )
    return new_x.detach()

# ---------------------------------------------------------
# Placeholder decode function
# ---------------------------------------------------------

def decode_embeddings_to_sequences(emb: np.ndarray, reference_seq: str) -> List[str]:
    """**Toy**: return 1‑AA mutants nearest in latent *cosine* distance."""
    # In practice: nearest neighbour in a pre‑computed (seq, emb) database, or
    # differentiable decoder / language model prompting.
    # Here: just mutate one site towards Alanine per candidate.
    seqs = []
    for i in range(len(emb)):
        pos = i % len(reference_seq)
        seqs.append(reference_seq[:pos] + "A" + reference_seq[pos + 1 :])
    return seqs

# ---------------------------------------------------------
# Argument parsing / main loop
# ---------------------------------------------------------

def main(args):
    df = pd.read_csv(args.seed_data)
    if {args.expr_col, args.bind_col, "sequence"}.difference(df.columns):
        raise ValueError("Input CSV must contain 'sequence', expr_col, bind_col columns")

    api = Evo2Client(args.api_key)

    # Embed sequences (cache to avoid re‑embedding)
    if not os.path.exists(args.embed_cache):
        emb = api.embed(df["sequence"].tolist())
        np.save(args.embed_cache, emb)
    else:
        emb = np.load(args.embed_cache)

    emb_t = torch.from_numpy(emb)

    # Replace NaNs for GP training; BoTorch requires finite targets.
    y = df[[args.expr_col, args.bind_col]].to_numpy(float)
    mask = ~np.isnan(y)
    # Simple imputation: NaN → column min
    col_min = np.nanmin(y, axis=0)
    y[~mask] = col_min[np.where(~mask)[1]]
    y_t = torch.from_numpy(y)

    model = _fit_gp(emb_t, y_t)

    ref_point = torch.min(y_t, dim=0).values - 0.1  # conservative ref

    for it in range(args.iterations):
        new_emb_t = propose_embeddings(model, ref_point, args.batch_size)
        new_emb = new_emb_t.numpy()
        new_seqs = decode_embeddings_to_sequences(new_emb, df["sequence"].iloc[0])

        out = pd.DataFrame({
            "sequence": new_seqs,
            "expr_pred": np.nan,  # to be filled after measurement
            "bind_pred": np.nan,
        })
        fname = f"proposals_round_{it + 1}.csv"
        out.to_csv(fname, index=False)
        print(f"[Iter {it + 1}] Wrote {fname} – please measure binding & expression, then append to seed CSV and rerun.")
        break  # demo: one round only


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed-data", required=True, help="CSV with sequence, expr, bind cols")
    p.add_argument("--expr-col", default="expr")
    p.add_argument("--bind-col", default="bind")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--api-key", default=os.getenv("NVIDIA_API_KEY"))
    p.add_argument("--embed-cache", default="embeddings.npy")
    args = p.parse_args()

    main(args)
