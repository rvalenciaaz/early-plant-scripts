#!/usr/bin/env python3
"""
Download Ensembl Fungi peptide FASTA files and map their translation IDs
to UniProt accessions, writing one TSV per species.

Requires: biopython, requests
"""

import os
import gzip
import ftplib
import requests
from pathlib import Path
from Bio import SeqIO

# ──────────────────────────────── CONFIG ────────────────────────────────
FTP_HOST       = "ftp.ensemblgenomes.ebi.ac.uk"
BASE_DIR       = "/pub/fungi/release-61/fasta"
LOCAL_GENOMES  = Path("ensemblfunga_genomes")   # where your *.dna.toplevel.fa.gz live
LOCAL_PEP      = Path("ensemblfunga_pep")       # peptide FASTA cache
OUTPUT_DIR     = Path("uniprot_mappings")       # TSV output
REST_SERVER    = "https://rest.ensembl.org"
HEADERS        = {"Accept": "application/json"}  # ← correct header
# ────────────────────────────────────────────────────────────────────────

LOCAL_PEP.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def list_species(genomes_dir: Path) -> list[str]:
    """Return a sorted list of species names inferred from local genome files."""
    return sorted({
        fn.name.split(".", 1)[0].lower()
        for fn in genomes_dir.glob("*.dna.toplevel.fa.gz")
    })

def download_pep(ftp: ftplib.FTP, species: str) -> Path | None:
    """Grab the species’ *.pep.all.fa.gz file and cache it locally."""
    pep_dir = f"{BASE_DIR}/{species}/pep"
    try:
        ftp.cwd(pep_dir)
    except ftplib.error_perm:
        print(f"⚠️  No protein dir for {species}")
        return None

    pep_files = [f for f in ftp.nlst() if f.endswith(".pep.all.fa.gz")]
    if not pep_files:
        print(f"⚠️  No *.pep.all.fa.gz for {species}")
        return None

    local_path = LOCAL_PEP / pep_files[0]
    if not local_path.exists():
        print(f"  ↳ downloading {pep_files[0]}")
        with open(local_path, "wb") as fh:
            ftp.retrbinary(f"RETR {pep_files[0]}", fh.write)
    return local_path

def parse_translation_ids(fasta_gz: Path, limit: int | None = None) -> list[str]:
    """Parse translation stable IDs from a peptide FASTA (optionally capped)."""
    ids = []
    with gzip.open(fasta_gz, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            ids.append(rec.id)
            if limit and len(ids) >= limit:
                break
    return ids

def map_uniprot(translation_id: str) -> list[str]:
    """Return all UniProt accessions linked to an Ensembl translation."""
    url = f"{REST_SERVER}/xrefs/id/{translation_id}"
    params = {"all_levels": 1}  # include transcript / gene links if any
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()

    # Keep any db that starts with 'UniProt'
    return [
        hit["primary_id"]
        for hit in r.json()
        if hit["dbname"].startswith("UniProt")
    ]

def process_species(ftp: ftplib.FTP, species: str, test_n: int = 10) -> None:
    """Download peptide file, map first N proteins, write TSV."""
    print(f"\n⏳ Processing {species}")
    pep_file = download_pep(ftp, species)
    if pep_file is None:
        return

    prot_ids = parse_translation_ids(pep_file, limit=test_n)
    print(f"  → Mapping {len(prot_ids)} proteins")

    mappings: list[tuple[str, list[str]]] = []
    for pid in prot_ids:
        try:
            ups = map_uniprot(pid)
        except Exception as exc:
            print(f"    ⚠️  {pid}: {exc}")
            ups = []
        mappings.append((pid, ups))

    out_path = OUTPUT_DIR / f"{species}_mapping.tsv"
    with open(out_path, "w") as out:
        out.write("EnsemblID\tUniProtKB\n")
        for pid, ups in mappings:
            out.write(f"{pid}\t{';'.join(ups)}\n")
    print(f"  ✔ Wrote {out_path}")

def main() -> None:
    with ftplib.FTP(FTP_HOST) as ftp:
        ftp.login()
        for sp in list_species(LOCAL_GENOMES):
            process_species(ftp, sp)

if __name__ == "__main__":
    main()
