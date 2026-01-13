import os
import re
import json
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import joblib

random_state = 42

base_dir = os.getcwd()
preproc_dir = os.path.join(base_dir, "outputs", "preprocessing")
features_dir = os.path.join(base_dir, "outputs", "features")

os.makedirs(features_dir, exist_ok=True)

train_path = os.path.join(preproc_dir, "malicious_phish_train.csv")
val_path = os.path.join(preproc_dir, "malicious_phish_val.csv")
test_path = os.path.join(preproc_dir, "malicious_phish_test.csv")
label_map_path = os.path.join(preproc_dir, "label_mapping.json")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

with open(label_map_path, "r", encoding="utf-8") as f:
    label_to_id = json.load(f)

suspicious_tlds = {
    "zip","xyz","top","gq","cf","tk","ml","ga","click","link","work",
    "info","ru","cn","quest","icu","party","science","biz"
}

keywords = [
    "login","signin","verify","update","account","bank","secure",
    "free","win","bonus","gift","paypal","confirm"
]

def shannon_entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    probs = np.array(list(counts.values()), dtype=float) / len(s)
    return float(-np.sum(probs * np.log2(probs)))

def extract_url_features(url):
    if not isinstance(url, str):
        url = ""
    u = url.strip()
    if not re.match(r"^\w+://", u):
        u = "http://" + u
    parsed = urlparse(u)
    host = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    url_len = len(url)
    host_len = len(host)
    path_len = len(path)
    query_len = len(query)

    count_digits = sum(c.isdigit() for c in url)
    count_letters = sum(c.isalpha() for c in url)
    count_symbols = sum(not c.isalnum() for c in url)
    count_dots = url.count(".")
    count_dashes = url.count("-")
    count_underscores = url.count("_")
    count_slashes = url.count("/")
    count_question = url.count("?")
    count_equal = url.count("=")
    count_at = url.count("@")
    count_percent = url.count("%")

    num_params = query.count("&") + (1 if query else 0)
    num_subdomains = max(host.count(".") - 1, 0)
    path_depth = path.count("/")

    has_ip = 1 if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", host) else 0
    has_https = 1 if parsed.scheme.lower() == "https" or "https" in url.lower() else 0
    has_http = 1 if "http" in url.lower() else 0

    url_len_safe = url_len if url_len > 0 else 1
    digit_ratio = count_digits / url_len_safe
    letter_ratio = count_letters / url_len_safe
    symbol_ratio = count_symbols / url_len_safe

    tld = ""
    if host and "." in host:
        tld = host.split(".")[-1].split(":")[0].lower()
    is_suspicious_tld = 1 if tld in suspicious_tlds else 0

    entropy = shannon_entropy(url)

    lower_url = url.lower()
    feats = {
        "url_len": url_len,
        "host_len": host_len,
        "path_len": path_len,
        "query_len": query_len,
        "count_digits": count_digits,
        "count_letters": count_letters,
        "count_symbols": count_symbols,
        "count_dots": count_dots,
        "count_dashes": count_dashes,
        "count_underscores": count_underscores,
        "count_slashes": count_slashes,
        "count_question": count_question,
        "count_equal": count_equal,
        "count_at": count_at,
        "count_percent": count_percent,
        "num_params": num_params,
        "num_subdomains": num_subdomains,
        "path_depth": path_depth,
        "digit_ratio": digit_ratio,
        "letter_ratio": letter_ratio,
        "symbol_ratio": symbol_ratio,
        "entropy": entropy,
        "has_ip": has_ip,
        "has_https": has_https,
        "has_http": has_http,
        "is_suspicious_tld": is_suspicious_tld,
    }
    for kw in keywords:
        feats[f"kw_{kw}"] = 1 if kw in lower_url else 0
    return feats

def build_numeric_features(df):
    feats = df["url"].apply(extract_url_features).apply(pd.Series)
    return feats

X_train_num = build_numeric_features(train_df)
X_val_num = build_numeric_features(val_df)
X_test_num = build_numeric_features(test_df)

numeric_feature_names = X_train_num.columns.tolist()

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num.values)
X_val_num_scaled = scaler.transform(X_val_num.values)
X_test_num_scaled = scaler.transform(X_test_num.values)

X_train_num_scaled_sparse = sparse.csr_matrix(X_train_num_scaled)
X_val_num_scaled_sparse = sparse.csr_matrix(X_val_num_scaled)
X_test_num_scaled_sparse = sparse.csr_matrix(X_test_num_scaled)

tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    max_df=0.999,
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(train_df["url"].astype(str))
X_val_tfidf = tfidf.transform(val_df["url"].astype(str))
X_test_tfidf = tfidf.transform(test_df["url"].astype(str))

X_train_all = sparse.hstack([X_train_num_scaled_sparse, X_train_tfidf], format="csr")
X_val_all = sparse.hstack([X_val_num_scaled_sparse, X_val_tfidf], format="csr")
X_test_all = sparse.hstack([X_test_num_scaled_sparse, X_test_tfidf], format="csr")

y_train = train_df["label"].values
y_val = val_df["label"].values
y_test = test_df["label"].values

sparse.save_npz(os.path.join(features_dir, "X_train_all.npz"), X_train_all)
sparse.save_npz(os.path.join(features_dir, "X_val_all.npz"), X_val_all)
sparse.save_npz(os.path.join(features_dir, "X_test_all.npz"), X_test_all)

np.save(os.path.join(features_dir, "y_train.npy"), y_train)
np.save(os.path.join(features_dir, "y_val.npy"), y_val)
np.save(os.path.join(features_dir, "y_test.npy"), y_test)

X_train_num.to_csv(os.path.join(features_dir, "X_train_numeric.csv"), index=False)
X_val_num.to_csv(os.path.join(features_dir, "X_val_numeric.csv"), index=False)
X_test_num.to_csv(os.path.join(features_dir, "X_test_numeric.csv"), index=False)

joblib.dump(scaler, os.path.join(features_dir, "numeric_scaler.joblib"))
joblib.dump(tfidf, os.path.join(features_dir, "tfidf_vectorizer.joblib"))

meta = {
    "numeric_feature_names": numeric_feature_names,
    "n_numeric_features": int(len(numeric_feature_names)),
    "n_tfidf_features": int(X_train_tfidf.shape[1]),
    "X_train_shape": list(X_train_all.shape),
    "X_val_shape": list(X_val_all.shape),
    "X_test_shape": list(X_test_all.shape),
    "label_mapping": label_to_id,
}
with open(os.path.join(features_dir, "features_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Feature engineering complete.")
print(json.dumps(meta, indent=2))
