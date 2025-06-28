#!/usr/bin/env python3
"""
simulate_metagenomes_mess.py – version 2.1
* uses MeSS instead of NanoSim
* FIX: no tax_id column → avoids IntCastingNaNError
"""
import argparse, gzip, itertools, os, random, re, subprocess, sys, textwrap
from pathlib import Path
from typing import List, Tuple

FA_RX = re.compile(r"(\.fa(sta)?|\.fna)(\.gz)?$|\.dna\.toplevel(\.gz)?$", re.I)

def list_genomes(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if FA_RX.search(p.name))

def clean_tag(fp: Path) -> str:
    return re.sub(r"[^\w]+", "_", FA_RX.sub("", fp.name)).strip("_")

def write_simulate_tsv(plant_path: Path, fungus_path: Path, outdir: Path) -> Path:
    sample = f"{clean_tag(plant_path)}__{clean_tag(fungus_path)}"
    tsv    = outdir / "simulate_input.tsv"
    with tsv.open("w") as fh:
        fh.write("fasta\tpath\tnb\tcov_sim\tsample\n")
        fh.write(f"{clean_tag(plant_path)}\t{plant_path.resolve()}\t1\t50\t{sample}\n")
        fh.write(f"{clean_tag(fungus_path)}\t{fungus_path.resolve()}\t1\t50\t{sample}\n")
    return tsv

def run_mess(input_tsv: Path, outdir: Path, tech: str, threads: int, deploy: str):
    cmd = ["mess", "simulate", "-i", str(input_tsv), "-o", str(outdir),
           "--tech", tech, "--threads", str(threads), "--sdm", deploy]
    print("🟢", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def main(argv=None):
    p = argparse.ArgumentParser(description="Simulate plant × fungus metagenomes with MeSS")
    p.add_argument("--plant_dir", default="plant-multi-species-genomes", type=Path)
    p.add_argument("--fungi_dir", default="ensemblfunga_genomes", type=Path)
    p.add_argument("-o", "--out_dir", default="mess_out", type=Path)
    p.add_argument("--tech", choices=["nanopore", "illumina", "pacbio"], default="nanopore")
    p.add_argument("-t", "--threads", type=int, default=4)
    p.add_argument("--sdm", choices=["conda", "apptainer"], default="conda")
    p.add_argument("--test", action="store_true", help="only one random pair")
    args = p.parse_args(argv)

    plants = list_genomes(args.plant_dir)
    fungi  = list_genomes(args.fungi_dir)
    if not plants: sys.exit(f"No genomes in {args.plant_dir}")
    if not fungi:  sys.exit(f"No genomes in {args.fungi_dir}")

    pairs: List[Tuple[Path, Path]] = (
        [(random.choice(plants), random.choice(fungi))] if args.test
        else list(itertools.product(plants, fungi))
    )

    for plant_fp, fungus_fp in pairs:
        pair_dir = args.out_dir / f"{clean_tag(plant_fp)}__{clean_tag(fungus_fp)}"
        if (pair_dir / "fastq").exists():
            print(f"⚠️  {pair_dir.name} already simulated – skipping")
            continue
        pair_dir.mkdir(parents=True, exist_ok=True)
        sim_tsv = write_simulate_tsv(plant_fp, fungus_fp, pair_dir)
        run_mess(sim_tsv, pair_dir, args.tech, args.threads, args.sdm)

if __name__ == "__main__":
    main()
