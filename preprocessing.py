import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

random_state = 42
base_dir = os.getcwd()
input_path = os.path.join(base_dir, "malicious_phish.csv")
outputs_dir = os.path.join(base_dir, "outputs")
preproc_dir = os.path.join(outputs_dir, "preprocessing")

os.makedirs(preproc_dir, exist_ok=True)

df = pd.read_csv(input_path)

df.columns = [c.strip().lower() for c in df.columns]
if "url" not in df.columns or "type" not in df.columns:
    raise ValueError("Input file must contain 'url' and 'type' columns")

df = df[["url", "type"]]

df["url"] = df["url"].astype(str)
df["type"] = df["type"].astype(str)

df["url"] = df["url"].str.replace(r"[\r\n\t]", " ", regex=True)
df["url"] = df["url"].str.strip()
df["type"] = df["type"].str.strip()

missing_like = {"", "nan", "none", "null", "na", "n/a", "?"}
df = df[~df["url"].str.lower().isin(missing_like)]
df = df[~df["type"].str.lower().isin(missing_like)]

df = df.dropna(subset=["url", "type"])

df["url"] = df["url"].str.replace(r"\s+", "", regex=True)
df["url"] = df["url"].str.lower()
df["type"] = df["type"].str.lower()

df = df[~df["url"].str.contains(r"[^ -~]", regex=True)]

df["url_len"] = df["url"].str.len()
df = df[df["url_len"] >= 5]
df = df[df["url_len"] <= 2048]

valid_labels = ["benign", "phishing", "defacement", "malware"]
df = df[df["type"].isin(valid_labels)]

label_counts_per_url = df.groupby("url")["type"].nunique()
ambiguous_urls = label_counts_per_url[label_counts_per_url > 1].index
df = df[~df["url"].isin(ambiguous_urls)]

df = df.drop_duplicates(subset=["url"])

df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

label_order = sorted(df["type"].unique())
label_to_id = {label: i for i, label in enumerate(label_order)}
df["label"] = df["type"].map(label_to_id)

X = df[["url", "type", "label"]]
y = df["label"]

test_size = 0.15
val_size = 0.15
train_val_size = 1.0 - test_size
relative_val_size = val_size / train_val_size

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    stratify=y,
    random_state=random_state,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=relative_val_size,
    stratify=y_train_val,
    random_state=random_state,
)

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

clean_full = pd.concat([X_train, X_val, X_test]).reset_index(drop=True)

full_clean_path = os.path.join(preproc_dir, "malicious_phish_clean_full.csv")
train_path = os.path.join(preproc_dir, "malicious_phish_train.csv")
val_path = os.path.join(preproc_dir, "malicious_phish_val.csv")
test_path = os.path.join(preproc_dir, "malicious_phish_test.csv")
label_map_path = os.path.join(preproc_dir, "label_mapping.json")

clean_full.to_csv(full_clean_path, index=False)
X_train.to_csv(train_path, index=False)
X_val.to_csv(val_path, index=False)
X_test.to_csv(test_path, index=False)

with open(label_map_path, "w", encoding="utf-8") as f:
    json.dump(label_to_id, f, ensure_ascii=False, indent=2)

dist_full = clean_full["type"].value_counts().rename_axis("type").reset_index(name="count")
dist_train = X_train["type"].value_counts().rename_axis("type").reset_index(name="count")
dist_val = X_val["type"].value_counts().rename_axis("type").reset_index(name="count")
dist_test = X_test["type"].value_counts().rename_axis("type").reset_index(name="count")

dist_full.to_csv(os.path.join(preproc_dir, "class_distribution_full.csv"), index=False)
dist_train.to_csv(os.path.join(preproc_dir, "class_distribution_train.csv"), index=False)
dist_val.to_csv(os.path.join(preproc_dir, "class_distribution_val.csv"), index=False)
dist_test.to_csv(os.path.join(preproc_dir, "class_distribution_test.csv"), index=False)

summary = {
    "rows_clean_full": int(len(clean_full)),
    "rows_train": int(len(X_train)),
    "rows_val": int(len(X_val)),
    "rows_test": int(len(X_test)),
    "label_mapping": label_to_id,
    "class_distribution_full": dist_full.set_index("type")["count"].to_dict(),
    "class_distribution_train": dist_train.set_index("type")["count"].to_dict(),
    "class_distribution_val": dist_val.set_index("type")["count"].to_dict(),
    "class_distribution_test": dist_test.set_index("type")["count"].to_dict(),
    "min_url_len": int(clean_full["url"].str.len().min()),
    "max_url_len": int(clean_full["url"].str.len().max()),
}

with open(os.path.join(preproc_dir, "preprocessing_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Preprocessing complete.")
print("Summary:")
print(json.dumps(summary, indent=2))
