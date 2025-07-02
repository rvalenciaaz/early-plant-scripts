````markdown
# Evo2 Embeddings → K-Means, Mutational Series & t-SNE with Normalized Pseudo-Scoring (FASTA/FASTQ)

A self-contained notebook that accepts **FASTA _or_ FASTQ** inputs (plain or gzipped), slices sequences into 100 bp windows, embeds them with Helical AI’s Evo2 model, clusters, generates mutational series, computes a normalized pseudo-score, and visualizes everything in 2D (t-SNE). Results are saved as CSV and PNGs.

---

## 🚀 Installation

```bash
pip install --quiet helical-ai[models] biopython pandas scikit-learn matplotlib
````

---

## ⚙️ Parameters

```python
INPUT_PATH      = pathlib.Path("/workspace/GCF_001625215_1_ASM162521v1_genomic__Phaeosphaeria_nodorum_ASM14691v1_dna_toplevel.fq.gz")
KEEP            = 1000
MODEL_NAME      = "evo2-7b"
CLUSTERS        = 5
MUT_PER_REP     = 10
POINT_MUTATIONS = 3
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)
```

* **`INPUT_PATH`**: Path to input FASTA/FASTQ (may be gzipped).
* **`KEEP`**: Number of randomly sampled 100 bp slices to keep.
* **`MODEL_NAME`**: Evo2 model variant (e.g. `"evo2-7b"`).
* **`CLUSTERS`**: Number of k-means clusters.
* **`MUT_PER_REP`**: Number of mutants per representative slice.
* **`POINT_MUTATIONS`**: Number of substitutions per mutant.
* **`SEED`**: Random seed for reproducibility.

---

## 🛠 Helper Functions

```python
import gzip
import pathlib
import random
import time
from typing import List

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from helical.models.evo_2 import Evo2, Evo2Config

BASES = np.array(list("ACGT"))
SUPPORTED_FASTA = {".fa", ".fasta", ".fna"}
SUPPORTED_FASTQ = {".fq", ".fastq"}

def infer_format(path: pathlib.Path) -> str:
    parts = list(path.suffixes)
    if parts and parts[-1] == ".gz":
        parts.pop()
    ext = parts[-1].lower() if parts else None
    if ext in SUPPORTED_FASTA:
        return "fasta"
    if ext in SUPPORTED_FASTQ:
        return "fastq"
    raise ValueError(f"Unsupported file extension: {ext}")

def slide(seq: str, k: int = 100, stride: int = 100) -> List[str]:
    return [seq[i:i+k] for i in range(0, len(seq)-k+1, stride)]

def read_seqfile(path: pathlib.Path, k: int = 100, stride: int = 100) -> pd.DataFrame:
    fmt = infer_format(path)
    opener = gzip.open if path.suffix==".gz" or path.name.endswith(".gz") else open
    seqs = []
    with opener(path, "rt") as handle:
        for rec in SeqIO.parse(handle, fmt):
            seqs.extend(slide(str(rec.seq).upper(), k, stride))
    return pd.DataFrame(seqs, columns=["Sequence"])

def mean_pool(embs: List[np.ndarray]) -> np.ndarray:
    return np.stack([e.mean(axis=0) for e in embs])

def embed_with_evo2(df: pd.DataFrame, model_name: str, batch_size: int = 1) -> np.ndarray:
    cfg = Evo2Config(model_name=model_name, batch_size=batch_size)
    evo2 = Evo2(cfg)
    ds = evo2.process_data(df)
    out = evo2.get_embeddings(ds)
    return mean_pool(out["embeddings"])

def pick_representatives(emb: np.ndarray, labels: np.ndarray, km: KMeans) -> List[int]:
    reps = []
    for k in range(km.n_clusters):
        idxs = np.where(labels == k)[0]
        if idxs.size == 0:
            continue
        dists = np.linalg.norm(emb[idxs] - km.cluster_centers_[k], axis=1)
        reps.append(idxs[np.argmin(dists)])
    return reps

def mutate_seq(seq: str, n_mut: int = 1) -> str:
    arr = np.frombuffer(seq.encode(), dtype='|S1').astype(str)
    pos = np.random.choice(len(arr), size=n_mut, replace=False)
    for p in pos:
        choices = BASES[BASES != arr[p]]
        arr[p] = np.random.choice(choices)
    return ''.join(arr)
```

---

## 📈 Pipeline Steps

### 1 · Slice & Sample

```python
print(f"Reading slices from {INPUT_PATH}…")
df_orig = read_seqfile(INPUT_PATH)
assert len(df_orig) >= KEEP, f"Need ≥{KEEP} slices, found {len(df_orig)}"
df_orig = df_orig.sample(n=KEEP, random_state=SEED).reset_index(drop=True)
```

* Reads contigs/reads, breaks into 100 bp windows.
* Randomly retains `KEEP` windows.

---

### 2 · Embeddings (Original)

```python
print("Embedding original slices…")
emb_orig = embed_with_evo2(df_orig, MODEL_NAME)
```

* Produces an `(KEEP × 4096)` embedding matrix.

---

### 3 · K-Means Clustering

```python
print(f"Clustering k={CLUSTERS}…")
km = KMeans(n_clusters=CLUSTERS, random_state=SEED, n_init='auto')
labels = km.fit_predict(emb_orig)
df_orig['cluster'] = labels

cluster_sizes = pd.Series(labels).value_counts().to_dict()
largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
centroid_largest = km.cluster_centers_[largest_cluster]
print("Cluster sizes:", cluster_sizes)
```

* Assigns each slice to one of `CLUSTERS`.
* Identifies the largest cluster’s centroid.

---

### 4 · Compute Pseudo-Score

```python
def pseudo_score(emb: np.ndarray, cid: int) -> float:
    dist = np.linalg.norm(emb - centroid_largest)
    return (1.0 / cluster_sizes[cid]) * dist

df_orig['pseudo_score'] = [
    pseudo_score(emb_orig[i], labels[i])
    for i in range(len(df_orig))
]
```

* Score = distance to largest-cluster centroid × inverse cluster size.

---

### 5 · Mutational Series & Scoring

```python
rep_ids = pick_representatives(emb_orig, labels, km)
mut_seqs, mut_par, mut_scores = [], [], []

for rep in rep_ids:
    embp, cid = emb_orig[rep], labels[rep]
    for _ in range(MUT_PER_REP):
        m = mutate_seq(df_orig.loc[rep, 'Sequence'], POINT_MUTATIONS)
        mut_seqs.append(m)
        mut_par.append(rep)
        mut_scores.append(pseudo_score(embp, cid))

df_mut = pd.DataFrame({
    'Sequence': mut_seqs,
    'parent_idx': mut_par,
    'pseudo_score': mut_scores
})
```

* Generates `MUT_PER_REP` mutants per representative slice.
* Applies the same pseudo-score formula.

---

### 6 · Normalize Scores to \[0,1]

```python
all_scores = np.concatenate([
    df_orig.pseudo_score.values,
    df_mut.pseudo_score.values
])
min_s, max_s = all_scores.min(), all_scores.max()

df_orig['pseudo_score_norm'] = (df_orig.pseudo_score - min_s) / (max_s - min_s)
df_mut['pseudo_score_norm']  = (df_mut.pseudo_score  - min_s) / (max_s - min_s)
```

* Linear scaling of pseudo-scores across originals + mutants.

---

### 7 · Embeddings (Mutants)

```python
print("Embedding mutants…")
emb_mut = embed_with_evo2(df_mut, MODEL_NAME)
```

* Returns `(n_mutants × 4096)` embeddings.

---

### 8 · PCA + t-SNE on All Embeddings

```python
emb_all     = np.vstack([emb_orig, emb_mut])
labels_all  = np.concatenate([labels, -np.ones(len(emb_mut), dtype=int)])
parents_all = np.concatenate([-np.ones(len(emb_orig), dtype=int), df_mut.parent_idx.values])

pca  = PCA(n_components=50, random_state=SEED).fit_transform(emb_all)
tsne = TSNE(
    n_components=2, init='pca',
    learning_rate='auto', random_state=SEED
).fit_transform(pca)

coords = pd.DataFrame(tsne, columns=['tsne1','tsne2'])
coords['is_mut']            = labels_all < 0
coords['cluster']           = labels_all
coords['parent']            = parents_all
coords['pseudo_score_norm'] = np.concatenate([
    df_orig.pseudo_score_norm.values,
    df_mut.pseudo_score_norm.values
])
```

* Combines originals + mutants into one 2D embedding space.

---

### 9 · Plot: Clusters & Mutants

```python
fig, ax = plt.subplots(figsize=(6,5))
om, mm = ~coords.is_mut, coords.is_mut

ax.scatter(
    coords.tsne1[om], coords.tsne2[om],
    c=coords.cluster[om], s=20, alpha=0.8, label='Original'
)
ax.scatter(
    coords.tsne1[mm], coords.tsne2[mm],
    c=coords.parent[mm], marker='x', s=40,
    linewidths=1.2, label='Mutant'
)
ax.set(title='Sequence Distribution', xlabel='', ylabel='')
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(
    '/workspace/synthetic_metagenome_testing_cluster.png',
    dpi=600, bbox_inches='tight'
)
plt.close()
```

---

### 10 · Plot: Normalized Pseudo-Score

```python
fig, ax = plt.subplots(figsize=(6,5))
sc = ax.scatter(
    coords.tsne1, coords.tsne2,
    c=coords.pseudo_score_norm, s=30
)
plt.colorbar(sc, label='Normalized Pseudo-Score')
ax.set(
    title='Sequences Colored by Normalized Pseudo-Score',
    xlabel='', ylabel=''
)
plt.tight_layout()
plt.savefig(
    '/workspace/synthetic_metagenome_infectivity_norm.png',
    dpi=600, bbox_inches='tight'
)
plt.close()
```

---

### 11 · Save Results

```python
coords['Sequence'] = np.concatenate([
    df_orig.Sequence.values,
    df_mut.Sequence.values
])
coords.to_csv(
    'tsne_pseudo_infectivity_results.csv',
    index=False
)
print("Results saved to tsne_pseudo_infectivity_results.csv")
```

* **CSV**: `tsne_pseudo_infectivity_results.csv` with 2D coordinates, cluster, parent, normalized score, and sequence.
* **PNGs**: t-SNE cluster plot & normalized-score plot in `/workspace/`.

---

```
```
