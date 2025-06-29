````markdown
# Transcriptome Embeddings & Mutational Series – Helix-mRNA

This script takes raw FASTQ reads, computes Helix-mRNA embeddings, clusters the originals, projects them into 2D with t-SNE, generates a mutational series of the sequences, embeds those, and finally visualizes both originals and mutants. Outputs are saved as CSVs of t-SNE coordinates.

---

## 🚀 Installation

```bash
# Install required packages
pip install --quiet helical-ai[models] biopython pandas scikit-learn matplotlib
````

---

## ⚙️ Parameters

| Variable                  | Description                                                     |
| ------------------------- | --------------------------------------------------------------- |
| `TRANSCRIPTOME_FASTQ`     | Path to the input FASTQ (or gzipped FASTQ) file.                |
| `SAMPLE_SIZE`             | Number of reads to reservoir-sample (set `None` to read all).   |
| `SEED`                    | Random seed for reproducibility.                                |
| `MODEL_BATCH_SIZE`        | Batch size for Helix-mRNA embedding.                            |
| `MODEL_MAX_LENGTH`        | Maximum token length for model inputs.                          |
| `PERPLEXITY`              | t-SNE perplexity parameter.                                     |
| `K_manual`                | If set, overrides automatic k selection for k-means clustering. |
| **Mutational series**     |                                                                 |
| `MUT_N_ORIG`              | Number of original sequences to mutate (≤ `SAMPLE_SIZE`).       |
| `MUT_MAX_DIST`            | Maximum Hamming distance (number of substitutions) per mutant.  |
| `MUTS_PER_DIST`           | Number of mutants generated at each distance step.              |
| `MUT_NUCLEOTIDES`         | RNA alphabet for substitutions.                                 |
| `MUT_COLOR`, `MUT_MARKER` | Plot styling for mutants in t-SNE #2.                           |

---

## 🛠 Helper Functions

```python
def read_fastq(path: str, sample_size: int | None = None, seed: int = 0) -> pd.DataFrame:
    """
    Reservoir-sample `sample_size` reads from FASTQ(.gz). 
    If `sample_size` is None, reads entire file.
    Returns a DataFrame with a single column 'Sequence'.
    """
    ...
```

```python
def mean_pool(embs: np.ndarray) -> np.ndarray:
    """Averages per-token embeddings to get one vector per sequence."""
    return embs.mean(axis=1)
```

```python
def embed_with_helix_mrna(df: pd.DataFrame, device='cuda', batch_size=8, max_length=None) -> np.ndarray:
    """
    Converts DNA→RNA (T→U), instantiates HelixmRNA model,
    processes the DataFrame and returns mean-pooled embeddings.
    """
    ...
```

```python
def silhouette_best_k(X, ks=range(2,11), seed=0):
    """Computes silhouette scores over k in `ks` and returns the best k."""
    ...
```

```python
def mutate_sequence(seq: str, d: int, rng: random.Random) -> str:
    """
    Introduces exactly `d` random substitutions into `seq`,
    sampling positions without replacement and choosing new nucleotides.
    """
    ...
```

---

## 📈 Pipeline Steps

### 1. Read & Subsample

```python
df_orig = read_fastq(
    TRANSCRIPTOME_FASTQ,
    sample_size=SAMPLE_SIZE,
    seed=SEED
)
print(f"Sequences passed to embedding: {len(df_orig):,d}")
```

* **Reservoir sampling** ensures a uniform random sample of `SAMPLE_SIZE` reads.

---

### 2. Embed Originals

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

* **`embed_with_helix_mrna`** returns a NumPy array of shape `(n_sequences, embedding_dim)`.

---

### 3. Cluster Originals (k-means)

```python
X_std = StandardScaler().fit_transform(emb_orig)
best_k = int(K_manual) if K_manual is not None else silhouette_best_k(X_std, seed=SEED)
labels_orig = (
    np.zeros(len(X_std), dtype=int)
    if best_k == 1
    else KMeans(n_clusters=best_k, random_state=SEED).fit_predict(X_std)
)
print(f"Clusters: k = {best_k}")
```

* **Standardization** improves clustering.
* **Silhouette analysis** chooses the optimal k unless overridden.

---

### 4. PCA → t-SNE (#1, Originals Only)

```python
pca_orig  = PCA(n_components=50, random_state=SEED).fit_transform(emb_orig)
tsne_orig = TSNE(
    n_components=2, init='pca',
    learning_rate='auto', random_state=SEED,
    perplexity=PERPLEXITY
).fit_transform(pca_orig)

coords_orig = pd.DataFrame(tsne_orig, columns=['tsne1','tsne2'])
coords_orig['cluster'] = labels_orig

# Plot
plt.figure(figsize=(6,5))
scatter = plt.scatter(
    coords_orig['tsne1'], coords_orig['tsne2'],
    c=coords_orig['cluster'], cmap='tab20',
    s=15, alpha=0.8, linewidths=0
)
...
plt.show()
```

* Reduces embeddings to 50D with PCA before t-SNE for speed.
* Colors points by cluster.

---

### 5. Generate Mutational Series

```python
rng = random.Random(SEED)
pick_idx = rng.sample(range(len(df_orig)), k=min(MUT_N_ORIG, len(df_orig)))

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

* For each selected original, creates variants at distances 1…`MUT_MAX_DIST`.

---

### 6. Embed Mutants & Combine

```python
# Embed mutants
print("Embedding mutants …")
t1 = time.perf_counter()
emb_mut = embed_with_helix_mrna(
    df_mut,
    batch_size=MODEL_BATCH_SIZE,
    max_length=MODEL_MAX_LENGTH
)
print(f"Mutant embedding shape: {emb_mut.shape}  |  {time.perf_counter()-t1:.1f}s")

# Combine
emb_comb = np.concatenate([emb_orig, emb_mut], axis=0)
```

---

### 7. PCA → t-SNE (#2, Originals + Mutants)

```python
pca_comb  = PCA(n_components=50, random_state=SEED).fit_transform(emb_comb)
tsne_comb = TSNE(
    n_components=2, init='pca',
    learning_rate='auto', random_state=SEED,
    perplexity=PERPLEXITY
).fit_transform(pca_comb)

coords_comb = pd.DataFrame(tsne_comb, columns=['tsne1','tsne2'])
coords_comb['is_mut']   = np.r_[np.zeros(len(emb_orig), dtype=bool),
                                np.ones(len(emb_mut), dtype=bool)]
coords_comb['cluster']  = np.r_[labels_orig,
                                np.full(len(emb_mut), -1)]
coords_comb['mut_dist'] = np.r_[np.full(len(emb_orig), np.nan), mut_dist]
```

* Marks mutants vs. originals.
* Original clusters retained; mutants labeled `-1`.

---

### 8. Plot & Save Results

```python
# Plot t-SNE #2
plt.figure(figsize=(6,6))
# originals
scatter_orig = plt.scatter(...)

# mutants
plt.scatter(...)

plt.title('t-SNE #2 – originals + mutational series')
...
plt.show()

# Save CSVs
out_csv1 = pathlib.Path(TRANSCRIPTOME_FASTQ).with_suffix('.helix_mrna_tsne_clusters.csv')
coords_orig.to_csv(out_csv1, index=False)

out_csv2 = pathlib.Path(TRANSCRIPTOME_FASTQ).with_suffix('.helix_mrna_tsne_mutants.csv')
coords_comb.to_csv(out_csv2, index=False)

print('Saved →', out_csv1)
print('Saved →', out_csv2)
```

* Final visual overlay of originals (colored by cluster) and mutants (black ×).
* Exports coordinates for downstream analysis.

---

## 📂 Outputs

* `*.helix_mrna_tsne_clusters.csv` — t-SNE coordinates + cluster labels for originals.
* `*.helix_mrna_tsne_mutants.csv` — combined coordinates + mutation distances for all sequences.

---

```

*This Markdown can be included as a `README.md` in your GitHub repository to explain and document the pipeline clearly.*
```
