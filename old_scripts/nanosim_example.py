#!/usr/bin/env python
"""
Multi-species PTR-biased Nanopore read simulator
================================================

Combines:
  • Korem-et-al peak-to-trough coverage model (CoPTR Simulator class)
  • Meta-NanoSim empirical read-length & error profiles
to create one realistic FASTQ for a whole metagenome.

Requires:
  python >=3.9, numpy, pandas, pyfaidx, tqdm, joblib, biopython, nanosim>=3.2

--------------------------------------------------------------
Usage
-----
$ metagenome_ptr_long_read_sim.py \
      --config community.tsv           \
      --model  MetaNanoSim/Zymo_R9.4/  \
      --reads  300000                  \
      --out    metagenome.fastq.gz
--------------------------------------------------------------
"""
from __future__ import annotations
import argparse, gzip, pathlib, pickle, random
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from pyfaidx import Fasta
from tqdm import tqdm
import joblib

# ------------------------------------------------------------------
# 1.  Import Korem-style Simulator for PTR probability field
# ------------------------------------------------------------------
from coptr.tests.test_coptr import Simulator  # ships with CoPTR

BIN = 100                               # 100-bp bins (CoPTR default)
LEN_KDE = "training_aligned_reads.pkl"  # names inside Meta-NanoSim profile
ERR_MOD = "training_model_profile"      # ditto


# ------------------------------------------------------------------
# 2.  Long-read generator for *one* genome
# ------------------------------------------------------------------
class PTRLongReadSim:
    """Generate PTR-biased reads for a single circular genome."""
    def __init__(self, genome_fa: str, ptr: float, n_reads: int,
                 model_dir: str, ori_pos: float | None,
                 seed: int = 1):
        self.rng  = np.random.default_rng(seed)
        self.sim  = Simulator()
        self.ptr  = ptr
        self.n    = n_reads

        # --- genome sequence -------------------------------------------------
        fa = Fasta(genome_fa, read_ahead=1e8)
        self.seq  = str(next(iter(fa)).seq).upper()
        self.glen = len(self.seq)

        # --- Meta-NanoSim profiles ------------------------------------------
        self.kde  = joblib.load(pathlib.Path(model_dir, LEN_KDE))
        with open(pathlib.Path(model_dir, ERR_MOD), "rb") as fh:
            self.err_prof = pickle.load(fh)

        # --- PTR-weighted start distribution ---------------------------------
        _, _, ori, ter = self.sim.simulate_reads(
            1, ptr=ptr, ori_pos=ori_pos)                # 1 dummy read ⇒ probs
        self.probs  = self.sim.adjust_read_probs(ptr, ori, ter)
        self.cdf    = np.cumsum(self.probs)

    # ----------------------------------------------------------------
    def _rnd_start(self) -> int:
        u = self.rng.random()
        idx = np.searchsorted(self.cdf, u)
        return (idx * BIN + self.rng.integers(0, BIN)) % self.glen

    def _rnd_len(self) -> int:
        return max(100, int(self.kde.resample(1)[0]))

    def _slice_circ(self, start: int, length: int) -> str:
        end = start + length
        if end <= self.glen:
            return self.seq[start:end]
        return self.seq[start:] + self.seq[:end % self.glen]

    def _mutate(self, raw: str) -> Tuple[str, str]:
        """Introduce ONT errors, return (mutated_seq, qualities)."""
        return self.err_prof.introduce_errors(raw, rng=self.rng)

    # ----------------------------------------------------------------
    def emit(self, fhandle, prefix: str):
        """Write n_reads FASTQ records to an *already opened* handle."""
        for rid in range(self.n):
            s   = self._rnd_start()
            ln  = self._rnd_len()
            fwd = bool(self.rng.integers(0, 2))

            raw = self._slice_circ(s, ln)
            if not fwd: raw = str(Seq(raw).reverse_complement())

            seq_err, qual = self._mutate(raw)
            header = f"@{prefix}|r{rid}|pos={s}|len={ln}|ptr={self.ptr:.3f}"
            fhandle.write(
                f"{header}\n{seq_err}\n+\n{qual}\n".encode()
            )


# ------------------------------------------------------------------
# 3.  Main – iterate over genomes & stream to one FASTQ
# ------------------------------------------------------------------
def simulate_metagenome(table: pd.DataFrame,
                        model_dir: str,
                        total_reads: int,
                        out_fastq: str,
                        seed: int = 42):
    rng = random.Random(seed)

    # -- handle abundance vs. absolute counts ------------------------
    if table['abundance'].max() <= 1.0 + 1e-9:        # fractions
        table['reads'] = (table['abundance'] * total_reads).round().astype(int)
    else:                                             # explicit counts
        table['reads'] = table['abundance'].astype(int)
        total_reads = table['reads'].sum()

    # randomise species order so blocks don’t cluster in FASTQ
    table = table.sample(frac=1, random_state=seed).reset_index(drop=True)

    mode = "wb"
    with (gzip.open(out_fastq, mode) if out_fastq.endswith('.gz')
          else open(out_fastq, mode)) as fh:
        for idx, row in table.iterrows():
            gfa   = row['genome']
            ptr   = float(row['ptr'])
            ori   = None if pd.isna(row.get('ori_pos')) else float(row['ori_pos'])
            n     = int(row['reads'])
            sid   = pathlib.Path(gfa).stem

            print(f"▶  {sid}: {n} reads, PTR={ptr}, ori_pos={ori or 'rand'}")
            sim = PTRLongReadSim(gfa, ptr, n, model_dir, ori,
                                 seed=rng.randint(0, 2**31-1))
            sim.emit(fh, prefix=sid)

    print(f"\n✓ finished.  Wrote {total_reads:,} reads to {out_fastq}")


# ------------------------------------------------------------------
# 4.  CLI -----------------------------------------------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="TSV/CSV with columns: genome, abundance|reads, ptr, [ori_pos]")
    p.add_argument("--model",  required=True, help="Meta-NanoSim profile directory")
    p.add_argument("--reads",  type=int, default=0,
                   help="Total reads (ignored if abundance column already absolute)")
    p.add_argument("--out",    required=True, help="output FASTQ(.gz)")
    args = p.parse_args()

    ext = pathlib.Path(args.config).suffix.lower()
    table = (pd.read_csv(args.config, sep='\t' if ext in ['.tsv', '.txt'] else ',')
                 .dropna(subset=['genome', 'ptr', 'abundance']))
    simulate_metagenome(table, args.model, args.reads, args.out)


if __name__ == "__main__":
    cli()
