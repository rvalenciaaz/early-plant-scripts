#!/usr/bin/env python3
"""
Generate a mutational series of nanobody (VHH) sequences and reverse‑translate them to DNA.

Features
--------
* Loads protein sequences from a FASTA file.
* Creates all single‑amino‑acid mutants (optionally uses Evo 2 to score / suggest best variants).
* Reverse‑translates protein variants to DNA using a user‑selectable codon usage table (default: *E. coli* bias).
* Outputs: (1) CSV with metadata + sequences; (2) FASTA of DNA variants.
* Designed to run locally; can call Nvidia‑hosted Evo 2 API if NVIDIA_API_KEY is set.

Usage
-----
$ python nanobody_mut_series.py \
    --input-fasta seeds.faa \
    --output-csv mutants.csv \
    --output-fasta mutants.fna \
    --top-k 16 \
    --use-evo            # optional; scores with Evo 2 and keeps top‑k

Requirements
------------
- Python ≥3.10
- biopython
- pandas
- requests
- tqdm

Optional:
- optipyzer or dna-chisel (for advanced codon optimisation)

"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CODON USAGE TABLES
# -----------------------------------------------------------------------------
# Minimal *E. coli* K‑12 codon table – weighted most‑used codon first.
DEFAULT_CODON_TABLE: Dict[str, List[str]] = {
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "C": ["TGC", "TGT"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "F": ["TTT", "TTC"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
    "H": ["CAT", "CAC"],
    "I": ["ATT", "ATC", "ATA"],
    "K": ["AAA", "AAG"],
    "L": ["CTG", "TTA", "TTG", "CTC", "CTA", "CTT"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "P": ["CCG", "CCT", "CCC", "CCA"],
    "Q": ["CAA", "CAG"],
    "R": ["CGT", "CGC", "AGA", "AGG", "CGG", "CGA"],
    "S": ["TCG", "TCC", "TCT", "TCA", "AGC", "AGT"],
    "T": ["ACC", "ACT", "ACA", "ACG"],
    "V": ["GTG", "GTT", "GTC", "GTA"],
    "W": ["TGG"],
    "Y": ["TAT", "TAC"],
    "*": ["TAA", "TAG", "TGA"],
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def reverse_translate(aa_seq: str, codon_table: Dict[str, List[str]] | None = None) -> str:
    """Reverse‑translate an amino‑acid sequence to a DNA sequence.

    The *first* (most‑preferred) codon for each amino acid is used.
    """
    if codon_table is None:
        codon_table = DEFAULT_CODON_TABLE
    try:
        return "".join(codon_table[aa][0] for aa in aa_seq)
    except KeyError as exc:
        raise ValueError(f"Unknown amino‑acid code in sequence: {exc}") from None


def generate_single_mutants(aa_seq: str, alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> Iterable[tuple[str, int, str, str]]:
    """Yield all single‑amino‑acid substitutions for *aa_seq*.

    Yields tuples of (mutated_sequence, position_index, wt_residue, mut_residue).
    Positions are 0‑based internally; +1 later for human‑readable output.
    """
    for idx, wt in enumerate(aa_seq):
        for aa in alphabet:
            if aa != wt:
                yield aa_seq[: idx] + aa + aa_seq[idx + 1 :], idx, wt, aa


# -----------------------------------------------------------------------------
# Evo 2 interface (Nvidia Hosted API). Docs: https://developer.nvidia.com/nim
# -----------------------------------------------------------------------------

def _request_evo(endpoint: str, payload: dict, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def evo_score_ll(sequences: Sequence[str], model: str = "evo-2-base", api_key: str | None = None) -> List[float]:
    """Return log‑likelihoods for *sequences* using Evo 2.* API quota permitting."""
    if api_key is None:
        raise RuntimeError("Set NVIDIA_API_KEY env‑var or pass api_key explicitly to use Evo scoring.")

    # Truncate very long sequences (>8 k) to keep within API limits.
    clean = [s[:8000] for s in sequences]
    payload = {"model": model, "sequences": clean}
    data = _request_evo("https://api.nvidia.com/v1/genai/evo2/score", payload, api_key)
    return data["log_likelihoods"]


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def process_record(rec, top_k: int, use_evo: bool, api_key: str | None):
    protein = str(rec.seq).upper()
    mutants = list(generate_single_mutants(protein))

    if use_evo:
        sequences = [m[0] for m in mutants]
        ll = evo_score_ll(sequences, api_key=api_key)
        scored = sorted(zip(mutants, ll), key=lambda x: x[1], reverse=True)[:top_k]
        selected = [m for (m, _) in scored]
    else:
        selected = mutants[:top_k]

    for mutated, idx, wt, mut in selected:
        dna = reverse_translate(mutated)
        yield {
            "parent_id": rec.id,
            "position": idx + 1,
            "wt": wt,
            "mut": mut,
            "protein_seq": mutated,
            "dna_seq": dna,
        }


def run(args):
    records = list(SeqIO.parse(args.input_fasta, "fasta"))
    all_rows = []

    pbar = tqdm(records, unit="seq", desc="Processing sequences")
    for rec in pbar:
        for row in process_record(rec, args.top_k, args.use_evo, args.api_key):
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False)

    # Write DNA variants to FASTA
    with open(args.output_fasta, "w") as handle:
        for _, row in df.iterrows():
            fasta_id = f"{row['parent_id']}_{row['position']}{row['wt']}>{row['mut']}"
            handle.write(f">{fasta_id}\n{row['dna_seq']}\n")

    print(f"\nGenerated {len(df)} variants → {args.output_csv}, {args.output_fasta}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate single‑mutant libraries for nanobody sequences")
    p.add_argument("--input-fasta", required=True, help="FASTA of seed nanobody protein sequences (AA)")
    p.add_argument("--output-csv", default="nanobody_mutants.csv", help="CSV metadata output")
    p.add_argument("--output-fasta", default="nanobody_mutants.fna", help="FASTA of DNA variants")
    p.add_argument("--top-k", type=int, default=64, help="Number of variants to keep per parent sequence")
    p.add_argument("--use-evo", action="store_true", help="Score mutants with Evo 2 and keep top‑k")
    p.add_argument("--api-key", default=os.getenv("NVIDIA_API_KEY"), help="Nvidia GenAI API key for Evo 2")
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
