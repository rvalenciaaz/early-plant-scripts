#!/usr/bin/env python3
"""
simulate_metagenomes.py
-----------------------

Simulate Meta-NanoSim metagenomes consisting of **one plant genome + one fungal
genome** per sample.

Modes
=====
• *All-vs-all* (default)  – every plant × every fungus combination
• *Test*  (--test)        – one random plant × fungus pair for a quick smoke-test

Key features
============
✓ Accepts .fa, .fasta, .fa.gz, .fasta.gz  
✓ Auto-generates genome_list.tsv, abundance.tsv **and dna_type_list.tsv**  
✓ Enforces every chromosome as *linear* by default (safe for multi-contig assemblies)  
✓ Works with the official **pre-trained Meta-NanoSim models**  
✓ Idempotent: skips a pair if the simulated read file already exists  
✓ Threading via -t/--threads; FASTQ output with --fastq

Example
-------
# full all-vs-all run (FASTA output)
./simulate_metagenomes.py -t 8

# quick test: one random pair, FASTQ output
./simulate_metagenomes.py --test --fastq
"""

import argparse
import gzip
import itertools
import os
import random
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple


# ───────────────────────────────── helpers ────────────────────────────────────
def list_genomes(folder: Path) -> List[Path]:
    """Return sorted list of genomes (*.fa[sta][.gz])."""
    return sorted(
        p
        for p in folder.glob("*")
        if re.search(r"\.fa(sta)?(\.gz)?$", p.name, re.IGNORECASE)
    )


def species_tag(fp: Path) -> str:
    """Strip extension and spaces → used as species name in NanoSim TSVs."""
    name = re.sub(r"\.fa(sta)?(\.gz)?$", "", fp.name, flags=re.IGNORECASE)
    return name.replace(" ", "_")


def first_header(fasta: Path) -> str:
    """Return the first FASTA/FASTQ header (chromosome ID)."""
    opener = gzip.open if fasta.suffix.endswith(".gz") else open
    with opener(fasta, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                return line[1:].split()[0]
    return "unknown_chr"


def write_genome_list(
    plant_tag: str, plant_file: Path, fungus_tag: str, fungus_file: Path, out: Path
) -> Path:
    gl = out / "genome_list.tsv"
    with gl.open("w") as f:
        f.write(f"{plant_tag}\t{plant_file}\n{fungus_tag}\t{fungus_file}\n")
    return gl


def write_abundance(
    plant_tag: str, fungus_tag: str, total_reads: int, out: Path
) -> Path:
    ab = out / "abundance.tsv"
    with ab.open("w") as f:
        f.write(f"Size\t{total_reads}\n{plant_tag}\t50\n{fungus_tag}\t50\n")
    return ab


def write_dna_type(
    plant_tag: str,
    plant_file: Path,
    fungus_tag: str,
    fungus_file: Path,
    out: Path,
) -> Path:
    dt = out / "dna_type_list.tsv"
    with dt.open("w") as f:
        f.write("Species\tChromosome\tDNA_type\n")
        f.write(f"{plant_tag}\t{first_header(plant_file)}\tlinear\n")
        f.write(f"{fungus_tag}\t{first_header(fungus_file)}\tlinear\n")
    return dt


def run_simulator(
    genome_list: Path,
    abundance: Path,
    dna_type: Path,
    model: Path,
    out_prefix: Path,
    threads: int,
    fastq: bool,
) -> None:
    cmd = [
        "simulator.py",
        "metagenome",
        "-gl",
        str(genome_list),
        "-a",
        str(abundance),
        "-dl",
        str(dna_type),
        "-c",
        str(model),
        "-o",
        str(out_prefix),
        "-t",
        str(threads),
    ]
    if fastq:
        cmd.append("--fastq")

    print("🟢", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ─────────────────────────────────── main ─────────────────────────────────────
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    parser.add_argument(
        "--plant_dir",
        type=Path,
        default="plant-multi-species-genomes",
        help="Folder with plant genomes",
    )
    parser.add_argument(
        "--fungi_dir",
        type=Path,
        default="ensemblfunga_genomes",
        help="Folder with fungal genomes",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=Path,
        default="metanosim_out",
        help="Output base directory",
    )
    parser.add_argument(
        "--model_prefix",
        type=Path,
        default=Path(
            "pre-trained_models/metagenome_ERR3152364_Even/training"
        ),
        help="Path to a Meta-NanoSim pre-trained model",
    )
    parser.add_argument(
        "-n",
        "--reads",
        type=int,
        default=200_000,
        help="Total reads per simulation (split 50/50)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=4, help="Threads for Meta-NanoSim"
    )
    parser.add_argument(
        "--fastq", action="store_true", help="Output FASTQ instead of FASTA"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only one random plant×fungus pair",
    )
    args = parser.parse_args(argv)

    plants = list_genomes(args.plant_dir)
    fungi = list_genomes(args.fungi_dir)

    if not plants:
        sys.exit(f"❌ No genomes found in {args.plant_dir}")
    if not fungi:
        sys.exit(f"❌ No genomes found in {args.fungi_dir}")

    pairs: List[Tuple[Path, Path]] = (
        [tuple(random.choice(plants) for _ in range(1)) + (random.choice(fungi),)]
        if args.test
        else list(itertools.product(plants, fungi))
    )

    for plant_file, fungus_file in pairs:
        plant_tag = species_tag(plant_file)
        fungus_tag = species_tag(fungus_file)
        pair_dir = args.out_dir / f"{plant_tag}__{fungus_tag}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        out_prefix = pair_dir / "simulated"
        already_done = out_prefix.with_suffix(".fastq").exists() or out_prefix.with_suffix(
            ".fasta"
        ).exists()
        if already_done:
            print(f"⚠️  {plant_tag} × {fungus_tag} already simulated, skipping.")
            continue

        gl = write_genome_list(plant_tag, plant_file, fungus_tag, fungus_file, pair_dir)
        ab = write_abundance(plant_tag, fungus_tag, args.reads, pair_dir)
        dt = write_dna_type(plant_tag, plant_file, fungus_tag, fungus_file, pair_dir)

        run_simulator(
            gl,
            ab,
            dt,
            args.model_prefix,
            out_prefix,
            args.threads,
            args.fastq,
        )


if __name__ == "__main__":
    main()
