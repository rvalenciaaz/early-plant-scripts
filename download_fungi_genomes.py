import requests
from bs4 import BeautifulSoup
import ftplib
import os

# 1) Fetch the PhytoPath pathogens page and extract the "Fungi" section
phyto_url = "https://phytopathdb.org/pathogens_eg/"
resp = requests.get(phyto_url)
resp.raise_for_status()  # abort if page fetch fails :contentReference[oaicite:0]{index=0}

# isolate everything up to the "Protists" heading
html_fungi = resp.text.split("## Protists")[0]

soup = BeautifulSoup(html_fungi, 'html.parser')
# species links all have hrefs like "/content/Alternaria_alternata"
species_links = [
    a for a in soup.find_all('a', href=True)
    if a['href'].startswith('/content/')
]

# extract the displayed binomial names
species_names = [a.text.strip() for a in species_links]

# 2) Map each species name → EnsemblFungi FTP directory:
#    lowercase, remove dots, replace spaces with underscores
species_dirs = [
    name.lower().replace('.', '').replace(' ', '_')
    for name in species_names
]

# 3) Connect to Ensembl Genomes FTP and download genomes
FTP_HOST = 'ftp.ensemblgenomes.org'
BASE_DIR = '/pub/fungi/current/fasta'   # current release symlink :contentReference[oaicite:1]{index=1}

out_dir = 'ensemblfunga_genomes'
os.makedirs(out_dir, exist_ok=True)

with ftplib.FTP(FTP_HOST) as ftp:
    ftp.login()  # anonymous
    for sp in species_dirs:
        sp_path = f"{BASE_DIR}/{sp}/dna"
        try:
            ftp.cwd(sp_path)
        except ftplib.error_perm:
            print(f"✗ no directory for {sp}")
            continue

        # list files and pick the top-level genome FASTA
        files = ftp.nlst()
        fasta = [f for f in files if f.endswith('dna.toplevel.fa.gz')]
        if not fasta:
            print(f"✗ no dna.toplevel.fa.gz for {sp}")
        else:
            fname = fasta[0]
            local = os.path.join(out_dir, fname)
            with open(local, 'wb') as lf:
                ftp.retrbinary(f"RETR {fname}", lf.write)
            print(f"✔ downloaded {fname} for {sp}")

        # return to the base before next loop
        ftp.cwd(BASE_DIR)
