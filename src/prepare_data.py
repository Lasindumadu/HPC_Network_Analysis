import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATASET_PATH = "./dataset/"
os.makedirs("./data", exist_ok=True)

print("Loading CICIDS-2017 dataset...")
frames = []
for fname in sorted(os.listdir(DATASET_PATH)):
    if fname.endswith(".csv"):
        print(f"  Reading: {fname}")
        frames.append(pd.read_csv(os.path.join(DATASET_PATH, fname), low_memory=False))

data = pd.concat(frames, ignore_index=True)
data.columns = data.columns.str.strip()
print(f"Full dataset shape: {data.shape}")

# Extract label as numpy array immediately - never touch dataframe again
label_col = [c for c in data.columns if c.lower() == 'label'][0]
y_full = (data[label_col].str.strip() != "BENIGN").astype(int).to_numpy()
data.drop(columns=[label_col], inplace=True)
print(f"Benign: {(y_full==0).sum()}  Attack: {(y_full==1).sum()}")

# Stratified 12% sampling using indices
rng = np.random.RandomState(42)
benign_idx  = np.where(y_full == 0)[0]
attack_idx  = np.where(y_full == 1)[0]
sampled_benign  = rng.choice(benign_idx,  size=int(len(benign_idx)  * 0.12), replace=False)
sampled_attack  = rng.choice(attack_idx,  size=int(len(attack_idx)  * 0.12), replace=False)
sampled_idx = np.concatenate([sampled_benign, sampled_attack])
sampled_idx.sort()

data = data.iloc[sampled_idx].reset_index(drop=True)
y    = y_full[sampled_idx].copy()
print(f"After 12% sampling: {len(data)} rows  (benign={( y==0).sum()}  attack={(y==1).sum()})")

# Clean inf/NaN — track valid rows
data.replace([np.inf, -np.inf], np.nan, inplace=True)
valid = data.notna().all(axis=1).to_numpy()
data = data[valid].reset_index(drop=True)
y    = y[valid]
print(f"After cleaning inf/NaN: {len(data)} rows")

# Keep only numeric feature columns
numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
data = data[numeric_cols]
print(f"Numeric features: {len(numeric_cols)}")

# Drop highly correlated features
corr   = data.corr().abs()
upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
data.drop(columns=to_drop, inplace=True)
final_cols = list(data.columns)
print(f"After correlation drop: {len(final_cols)} features")

# Final arrays
X = data.values.astype(np.float32)
y = y.astype(np.int32)
print(f"X: {X.shape}  y: {y.shape}  attack%: {y.mean()*100:.2f}%")
assert X.shape[0] == y.shape[0]

# Scale
X = StandardScaler().fit_transform(X).astype(np.float32)

# Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")

def save_bin(X, y, path):
    with open(path, "wb") as f:
        np.array([len(X), X.shape[1]], dtype=np.int32).tofile(f)
        X.tofile(f)
        y.tofile(f)
    print(f"  Saved {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

save_bin(X_tr, y_tr, "./data/train_data.bin")
save_bin(X_te, y_te, "./data/test_data.bin")

with open("./data/meta.txt", "w") as f:
    f.write(f"n_features={len(final_cols)}\n")
    f.write(f"train_rows={len(X_tr)}\n")
    f.write(f"test_rows={len(X_te)}\n")
    f.write(f"feature_names={','.join(final_cols)}\n")

print("Done!")
