# evo2_hybrid_pipeline.py
"""
End‑to‑end prototype for estimating a "Heterotic Potential Score" (HPS) for plant
hybrids using Evo 2 genomic language model outputs and a gradient‑boosting meta‑model.

Expected directory layout
-------------------------
project/
├── data/
│   ├── genomes/
│   │   ├── P1.fasta
│   │   ├── P2.fasta
│   │   └── ...
│   └── traits.csv            # columns: cross_id,parent1,parent2,yield,...
└── evo2_hybrid_pipeline.py   # (this file)

Quick start
-----------
# 1. train meta‑model
python evo2_hybrid_pipeline.py train \
       --genome_dir data/genomes \
       --traits_csv data/traits.csv \
       --out_dir artefacts

# 2. score all pairwise crosses
python evo2_hybrid_pipeline.py predict \
       --genome_dir data/genomes \
       --model artefacts/hps_model.txt \
       --out_csv artefacts/hps_scores.csv

Requirements (pip install ...)
------------------------------
biopython pandas numpy duckdb lightgbm tqdm scipy
# Evo 2: install from source → https://github.com/instadeepai/evo2
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import pathlib
import pickle
import textwrap
from dataclasses import dataclass

import duckdb as ddb
import numpy as np
import pandas as pd
from Bio import SeqIO
from lightgbm import LGBMRegressor
from scipy.spatial.distance import cosine
from tqdm import tqdm

# ‑‑‑‑‑ constants ‑‑‑‑‑
WINDOW_SIZE = 2_000  # bp per Evo 2 pass
STEP_SIZE = 2_000
EMBED_DIM = 2_048  # Evo 2 7B default; adjust if you use a different checkpoint

def sliding_windows(seq: str, size: int = WINDOW_SIZE, step: int = STEP_SIZE):
    """Yield (start, end, subseq) for sliding windows over *seq*."""
    for start in range(0, len(seq) - size + 1, step):
        yield start, start + size, seq[start : start + size]


@dataclass
class SeqFeatures:
    loglik_mean: float
    embedding_mean: np.ndarray  # EMBED_DIM


class Evo2Wrapper:
    """Thin wrapper around Evo 2 inference (FP16, deterministic)."""

    def __init__(self, checkpoint: str = "evo2-7b-fp16", device: str = "cuda"):
        try:
            from evo2 import Evo2  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "Evo 2 Python package not found. Follow installation instructions at"
                " https://github.com/instadeepai/evo2"
            ) from e
        self.model = Evo2(checkpoint, device=device)
        self._device = device

    def _infer_window(self, seq: str) -> tuple[float, np.ndarray]:
        """Return (mean loglikelihood, embedding)."""
        out = self.model.forward(seq, return_embedding=True)
        return float(out["loglikelihood"]), out["embedding"].cpu().numpy()

    def summarize_sequence(self, seq: str) -> SeqFeatures:
        """Aggregate Evo 2 outputs over sliding windows of *seq*."""
        lls = []
        embs = []
        for _, _, window in sliding_windows(seq):
            ll, emb = self._infer_window(window)
            lls.append(ll)
            embs.append(emb)
        return SeqFeatures(loglik_mean=np.mean(lls), embedding_mean=np.mean(embs, axis=0))


# ‑‑‑‑‑ feature engineering ‑‑‑‑‑

def make_virtual_f1(seq_a: str, seq_b: str) -> str:
    """Return IUPAC heterozygous consensus of two haploid sequences of equal length."""
    from Bio.Data import IUPACData  # maps frozenset("AC")→"M" etc.

    ambig_map = {{frozenset(v): k for k, v in IUPACData.ambiguous_dna_values.items()}}
    f1_chars = []
    for a, b in zip(seq_a, seq_b):
        if a == b:
            f1_chars.append(a)
        else:
            f1_chars.append(ambig_map.get(frozenset((a, b)), "N"))
    return "".join(f1_chars)


def cross_level_features(p1: SeqFeatures, p2: SeqFeatures, f1: SeqFeatures) -> dict[str, float]:
    emb_p1 = p1.embedding_mean
    emb_p2 = p2.embedding_mean
    emb_f1 = f1.embedding_mean

    return {
        "delta_ll": f1.loglik_mean - 0.5 * (p1.loglik_mean + p2.loglik_mean),
        "emb_dist": float(np.linalg.norm(emb_p1 - emb_p2)),
        "emb_cossim_f1_parents": 1.0 - cosine(emb_f1, (emb_p1 + emb_p2) / 2.0),
    }


# ‑‑‑‑‑ I/O helpers ‑‑‑‑‑

def load_parent_sequences(genome_dir: pathlib.Path) -> dict[str, str]:
    """Read all FASTA files in *genome_dir* into a dict id→sequence (upper‑case)."""
    seqs: dict[str, str] = {}
    for fasta in genome_dir.glob("*.fa*"):
        record = SeqIO.read(fasta, "fasta")
        seqs[fasta.stem] = str(record.seq).upper()
    if not seqs:
        raise FileNotFoundError(f"No FASTA files found in {genome_dir}")
    return seqs


def dump_duckdb(df: pd.DataFrame, parquet_path: pathlib.Path):
    parquet_path.parent.mkdir(exist_ok=True, parents=True)
    con = ddb.connect()
    con.execute("CREATE TABLE features AS SELECT * FROM df")
    con.execute(f"COPY features TO '{parquet_path}' (FORMAT 'parquet')")
    con.close()


# ‑‑‑‑‑ training & prediction logic ‑‑‑‑‑

def train_model(feature_table: pd.DataFrame, trait_table: pd.DataFrame, out_path: pathlib.Path):
    merged = trait_table.merge(feature_table, on=["parent1", "parent2"], how="inner")
    y = merged["yield"].values  # replace / expand trait name as needed
    X = merged[["delta_ll", "emb_dist", "emb_cossim_f1_parents"]].values

    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1)
    model.fit(X, y)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    model.booster_.save_model(out_path.as_posix())
    print(f"[✓] Saved LightGBM model → {out_path}")


def score_crosses(feature_table: pd.DataFrame, model_path: pathlib.Path) -> pd.DataFrame:
    from lightgbm import Booster

    booster = Booster(model_file=model_path.as_posix())
    X = feature_table[["delta_ll", "emb_dist", "emb_cossim_f1_parents"]].values
    feature_table = feature_table.copy()
    feature_table["HPS"] = booster.predict(X)
    return feature_table.sort_values("HPS", ascending=False)


# ‑‑‑‑‑ main orchestrator ‑‑‑‑‑

def compute_all_features(genome_dir: pathlib.Path, cache_path: pathlib.Path | None = None) -> pd.DataFrame:
    if cache_path and cache_path.exists():
        return pd.read_parquet(cache_path)

    print("[+] Loading parent genomes …")
    seqs = load_parent_sequences(genome_dir)

    evo = Evo2Wrapper()

    parent_feats: dict[str, SeqFeatures] = {}
    print("[+] Running Evo 2 on parents …")
    for pid, seq in tqdm(seqs.items(), desc="parents"):
        parent_feats[pid] = evo.summarize_sequence(seq)

    print("[+] Building and scoring virtual F1s …")
    records = []
    parent_ids = list(seqs.keys())
    for p1, p2 in tqdm(itertools.combinations(parent_ids, 2), desc="crosses"):
        f1_seq = make_virtual_f1(seqs[p1], seqs[p2])
        f1_feats = evo.summarize_sequence(f1_seq)
        feats = cross_level_features(parent_feats[p1], parent_feats[p2], f1_feats)
        records.append({"parent1": p1, "parent2": p2, **feats})

    df = pd.DataFrame.from_records(records)
    if cache_path:
        dump_duckdb(df, cache_path)
    return df


def cli():
    p = argparse.ArgumentParser(description="Evo 2 hybrid‑performance pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train meta‑model (LightGBM)")
    p_train.add_argument("--genome_dir", type=pathlib.Path, required=True)
    p_train.add_argument("--traits_csv", type=pathlib.Path, required=True)
    p_train.add_argument("--out_dir", type=pathlib.Path, required=True)

    # predict
    p_pred = sub.add_parser("predict", help="Score all pairwise crosses")
    p_pred.add_argument("--genome_dir", type=pathlib.Path, required=True)
    p_pred.add_argument("--model", type=pathlib.Path, required=True)
    p_pred.add_argument("--out_csv", type=pathlib.Path, required=True)

    args = p.parse_args()

    if args.cmd == "train":
        feature_cache = args.out_dir / "cross_features.parquet"
        df = compute_all_features(args.genome_dir, cache_path=feature_cache)
        traits = pd.read_csv(args.traits_csv)
        train_model(df, traits, args.out_dir / "hps_model.txt")

    elif args.cmd == "predict":
        df = compute_all_features(args.genome_dir, cache_path=None)
        scored = score_crosses(df, args.model)
        args.out_csv.parent.mkdir(exist_ok=True, parents=True)
        scored.to_csv(args.out_csv, index=False)
        print(f"[✓] Wrote ranked HPS table → {args.out_csv}")


if __name__ == "__main__":
    cli()
