````markdown
# Transcriptome Embeddings & Mutational Series – Helix-mRNA

This script takes raw FASTQ reads, computes Helix-mRNA embeddings, clusters the originals, projects them into 2D with t-SNE, generates a mutational series of the sequences, embeds those, and finally visualizes both originals and mutants. Outputs are saved as CSVs of t-SNE coordinates.

---

## 🚀 Installation

```bash
pip install --quiet helical-ai[models] biopython pandas scikit-learn matplotlib
````

---

## ⚙️ Parameters

```python
TRANSCRIPTOME_FASTQ = "/workspace/s3data/spottearly/data_srr/SRR12580257_1.fastq"
SAMPLE_SIZE         = 1000
SEED                = 42
MODEL_BATCH_SIZE    = 8
MODEL_MAX_LENGTH    = 1024
PERPLEXITY          = 30
K_manual            = None

MUT_N_ORIG          = 2000
MUT_MAX_DIST        = 3
MUTS_PER_DIST       = 1
MUT_NUCLEOTIDES     = "ACGU"
MUT_COLOR           = "black"
MUT_MARKER          = "x"
```

* **`TRANSCRIPTOME_FASTQ`**: Path to input FASTQ (or gzipped `.fastq.gz`).
* **`SAMPLE_SIZE`**: Number of reads to reservoir-sample (or `None` to read all).
* **`SEED`**: Random seed for reproducibility.
* **`MODEL_BATCH_SIZE`**, **`MODEL_MAX_LENGTH`**: Model inference settings.
* **`PERPLEXITY`**: t-SNE perplexity.
* **`K_manual`**: Override for k-means cluster count.
* **Mutational series** parameters control how many mutants are generated, their maximum Hamming distance, plotting color/marker, etc.

---

## 🛠 Helper Functions & Imports

```python
#!/usr/bin/env python
# coding: utf-8

import random, time, gzip, pathlib, sys, itertools
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig

# ────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)

def read_fastq(path: str,
               sample_size: int | None = None,
               seed: int = 0) -> pd.DataFrame:
    """Reservoir-sample `sample_size` reads from FASTQ(.gz); keep all if None."""
    opener = gzip.open if str(path).endswith(('.gz', '.gzip')) else open
    rng = random.Random(seed)

    if sample_size is None:
        with opener(path, 'rt') as handle:
            seqs = [str(rec.seq).upper()
                    for rec in SeqIO.parse(handle, 'fastq')]
        return pd.DataFrame(seqs, columns=['Sequence'])

    reservoir = []
    with opener(path, 'rt') as handle:
        for n, rec in enumerate(SeqIO.parse(handle, 'fastq')):
            seq = str(rec.seq).upper()
            if n < sample_size:
                reservoir.append(seq)
            else:
                j = rng.randint(0, n)
                if j < sample_size:
                    reservoir[j] = seq
    print(f"Streams read: {n+1:,d} | sample retained: {len(reservoir):,d}")
    return pd.DataFrame(reservoir, columns=['Sequence'])

def mean_pool(embs: np.ndarray) -> np.ndarray:
    return embs.mean(axis=1)

def embed_with_helix_mrna(df: pd.DataFrame,
                          device: str = 'cuda',
                          batch_size: int = 8,
                          max_length: int | None = None) -> np.ndarray:
    """Replace T→U, run HelixmRNA & return mean-pooled embeddings."""
    df = df.copy()
    df['Sequence'] = df['Sequence'].str.replace('T', 'U')
    cfg   = HelixmRNAConfig(device=device,
                            batch_size=batch_size,
                            max_length=max_length)
    helix = HelixmRNA(cfg)
    ds    = helix.process_data(df)
    out   = helix.get_embeddings(ds)
    return mean_pool(np.stack(out))

def silhouette_best_k(X, ks=range(2, 11), seed=0):
    """Return k (2–10) maximizing silhouette score, or 1 if too few points."""
    if len(X) < 2:
        return 1
    scores = [silhouette_score(
                X, KMeans(k, n_init='auto',
                          random_state=seed).fit_predict(X))
              for k in ks]
    return ks[int(np.argmax(scores))]

def mutate_sequence(seq: str,
                    d: int,
                    rng: random.Random) -> str:
    """Return a copy of `seq` with exactly `d` random substitutions."""
    seq_list = list(seq)
    L = len(seq_list)
    pos = rng.sample(range(L), k=min(d, L))
    for p in pos:
        orig = seq_list[p]
        alt_choices = [nuc for nuc in MUT_NUCLEOTIDES if nuc != orig]
        seq_list[p] = rng.choice(alt_choices)
    return ''.join(seq_list)
```

---

## 📈 Pipeline

### 1 · Read & Subsample

```python
df_orig = read_fastq(
    TRANSCRIPTOME_FASTQ,
    sample_size=SAMPLE_SIZE,
    seed=SEED
)
print(f"Sequences passed to embedding: {len(df_orig):,d}")
```

---

### 2 · Embed Originals

```python
print("Embedding originals with Helix-mRNA …")
t0 = time.perf_counter()
emb_orig = embed_with_helix_mrna(
    df_orig,
    batch_size=MODEL_BATCH_SIZE,
    max_length=MODEL_MAX_LENGTH
)
print(f"Original embedding shape: {emb_orig.shape}  |  {time.perf_counter()-t0:.1f}s")
```

---

### 3 · Cluster Originals

```python
X_std = StandardScaler().fit_transform(emb_orig)
best_k = int(K_manual) if K_manual is not None else silhouette_best_k(X_std, seed=SEED)
labels_orig = (np.zeros(len(X_std), dtype=int) if best_k == 1 else
               KMeans(n_clusters=best_k,
                      n_init='auto',
                      random_state=SEED).fit_predict(X_std))
print(f"Clusters: k = {best_k}")
```

---

### 4 · PCA → t-SNE #1

```python
pca_orig  = PCA(n_components=50, random_state=SEED).fit_transform(emb_orig)
tsne_orig = TSNE(n_components=2, init='pca',
                 learning_rate='auto',
                 random_state=SEED,
                 perplexity=PERPLEXITY).fit_transform(pca_orig)

coords_orig = pd.DataFrame(tsne_orig, columns=['tsne1','tsne2'])
coords_orig['cluster'] = labels_orig

plt.figure(figsize=(6,5))
scatter = plt.scatter(coords_orig['tsne1'], coords_orig['tsne2'],
                      c=coords_orig['cluster'], cmap='tab20',
                      s=15, alpha=0.8, linewidths=0)
plt.title('t-SNE #1 – originals only')
plt.xlabel('t-SNE-1'); plt.ylabel('t-SNE-2')
...
plt.show()
```

---

### 5 · Mutational Series

```python
rng = random.Random(SEED)
pick_idx = rng.sample(range(len(df_orig)),
                      k=min(MUT_N_ORIG, len(df_orig)))
mut_seqs, mut_parent_idx, mut_dist = [], [], []

for idx in pick_idx:
    parent = df_orig.iloc[idx]['Sequence']
    for d in range(1, MUT_MAX_DIST+1):
        for _ in range(MUTS_PER_DIST):
            mut_seqs.append(mutate_sequence(parent, d, rng))
            mut_parent_idx.append(idx)
            mut_dist.append(d)

df_mut = pd.DataFrame(mut_seqs, columns=['Sequence'])
print(f"Generated mutants: {len(df_mut):,d}")
```

---

### 6 · Embed Mutants

```python
print("Embedding mutants …")
t1 = time.perf_counter()
emb_mut = embed_with_helix_mrna(df_mut,
                                batch_size=MODEL_BATCH_SIZE,
                                max_length=MODEL_MAX_LENGTH)
print(f"Mutant embedding shape: {emb_mut.shape}  |  {time.perf_counter()-t1:.1f}s")
```

---

### 7 · Combine & t-SNE #2

```python
emb_comb = np.concatenate([emb_orig, emb_mut], axis=0)
pca_comb  = PCA(n_components=50, random_state=SEED).fit_transform(emb_comb)
tsne_comb = TSNE(n_components=2, init='pca',
                 learning_rate='auto',
                 random_state=SEED,
                 perplexity=PERPLEXITY).fit_transform(pca_comb)

coords_comb = pd.DataFrame(tsne_comb, columns=['tsne1','tsne2'])
coords_comb['is_mut']   = np.r_[np.zeros(len(emb_orig), dtype=bool),
                                np.ones(len(emb_mut),  dtype=bool)]
coords_comb['cluster']  = np.r_[labels_orig,
                                np.full(len(emb_mut), -1)]
coords_comb['mut_dist'] = np.r_[np.full(len(emb_orig), np.nan),
                                mut_dist]
```

---

### 8 · Plot t-SNE #2

```python
plt.figure(figsize=(6,6))
scatter_orig = plt.scatter(... originals ...)
plt.scatter(... mutants ...)
...
plt.show()
```

---

### 9 · Save Outputs

```python
out_csv1 = pathlib.Path(TRANSCRIPTOME_FASTQ)\
             .with_suffix('.helix_mrna_tsne_clusters.csv')
coords_orig.to_csv(out_csv1, index=False)

out_csv2 = pathlib.Path(TRANSCRIPTOME_FASTQ)\
             .with_suffix('.helix_mrna_tsne_mutants.csv')
coords_comb.to_csv(out_csv2, index=False)

print('Saved →', out_csv1)
print('Saved →', out_csv2)
```

* **CSV**: `.helix_mrna_tsne_clusters.csv` (originals).
* **CSV**: `.helix_mrna_tsne_mutants.csv` (originals + mutants).

---

```
```
