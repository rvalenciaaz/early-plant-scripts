#!/usr/bin/env python3
"""
From the per-species mapping TSVs produced by the Ensembl-to-UniProt
pipeline, fetch AlphaFold PDB files for every mapped UniProt accession.

Requires: requests
"""

from __future__ import annotations
import os
import csv
import time
import json
import hashlib
import requests
from pathlib import Path
from typing import Iterable, Set

# ─────────────────────────────── CONFIG ────────────────────────────────
MAPPING_DIR = Path("uniprot_mappings")                    # input folder
OUT_ROOT    = Path("alphafold_pdb")                       # where to save PDBs
API_BASE    = "https://alphafold.ebi.ac.uk/api/prediction"  # v1 endpoint:contentReference[oaicite:0]{index=0}
HEADERS     = {"Accept": "application/json"}
MAX_RETRY   = 3          # network retries per call
PAUSE_SECS  = 0.1        # courtesy delay between API hits
# ────────────────────────────────────────────────────────────────────────

OUT_ROOT.mkdir(exist_ok=True)

def mapping_files(mapping_dir: Path) -> Iterable[Path]:
    """Yield every *_mapping.tsv file except any global combined file."""
    for fn in mapping_dir.glob("*_mapping.tsv"):
        if fn.name != "all_species_mapping.tsv":
            yield fn

def extract_uniprot_ids(tsv_path: Path) -> Set[str]:
    """Return all non-empty UniProt accessions from a TSV."""
    ids: set[str] = set()
    with tsv_path.open() as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            for uid in row.get("UniProtKB", "").split(";"):  # ← correct header!
                uid = uid.strip()
                if uid:
                    ids.add(uid.split("-")[0])  # drop isoform suffix if present
    return ids

def safe_request(url: str, session: requests.Session) -> dict | None:
    """GET `url` with retries; return JSON dict or None on 404."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == MAX_RETRY:
                raise
            print(f"   ↻ retry {attempt}/{MAX_RETRY} for {url}: {exc}")
            time.sleep(PAUSE_SECS * 2)
    return None

def download_file(url: str, dest: Path, session: requests.Session) -> None:
    """Stream `url` → `dest`."""
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        dest_tmp = dest.with_suffix(".part")
        with dest_tmp.open("wb") as fh:
            for chunk in r.iter_content(8192):
                fh.write(chunk)
        dest_tmp.rename(dest)

def predictions_for(uid: str, cache_dir: Path, session: requests.Session) -> list[dict]:
    """
    Query AlphaFold API; cache raw JSON to save time if repeated.
    Cache key is SHA1(uniprotId).
    """
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha1(uid.encode()).hexdigest()[:12]
    cache_file = cache_dir / f"{key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    meta_url = f"{API_BASE}/{uid}"
    data = safe_request(meta_url, session)
    if data is None:
        return []

    cache_file.write_text(json.dumps(data))
    return data

def process_species(tsv_path: Path) -> None:
    species = tsv_path.stem.replace("_mapping", "")
    species_out = OUT_ROOT / species
    species_out.mkdir(exist_ok=True)

    print(f"\n🚀 {species}: reading mappings …", end="")
    uniprot_ids = extract_uniprot_ids(tsv_path)
    print(f" {len(uniprot_ids)} unique IDs")

    session = requests.Session()
    cache_dir = OUT_ROOT / ".cache"
    for uid in sorted(uniprot_ids):
        preds = predictions_for(uid, cache_dir, session)
        if not preds:
            print(f" ⚠️  no AlphaFold entry for {uid}")
            continue

        for pred in preds:
            pdb_url = pred.get("pdbUrl")
            if not pdb_url:
                continue
            fname   = os.path.basename(pdb_url)
            outpath = species_out / fname
            if outpath.exists():
                continue

            print(f"   ⬇︎ {uid} → {fname}")
            try:
                download_file(pdb_url, outpath, session)
            except Exception as exc:
                print(f"     ⚠️  failed: {exc}")
        time.sleep(PAUSE_SECS)   # small delay obeying EMBL-EBI etiquette

def main() -> None:
    for tsv in mapping_files(MAPPING_DIR):
        try:
            process_species(tsv)
        except Exception as exc:
            print(f"💥 error on {tsv.name}: {exc}")

    print("\n🎉 All downloads complete.")

if __name__ == "__main__":
    main()
