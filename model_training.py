import os
import json
import logging
import numpy as np
from scipy import sparse
import joblib

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)

random_state = 42

base_dir = os.getcwd()
outputs_dir = os.path.join(base_dir, "outputs")
features_dir = os.path.join(outputs_dir, "features")
models_dir = os.path.join(outputs_dir, "models")
os.makedirs(models_dir, exist_ok=True)

log_path = os.path.join(models_dir, "training_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("Loading feature matrices and labels")

X_train = sparse.load_npz(os.path.join(features_dir, "X_train_all.npz"))
X_val = sparse.load_npz(os.path.join(features_dir, "X_val_all.npz"))
X_test = sparse.load_npz(os.path.join(features_dir, "X_test_all.npz"))

y_train = np.load(os.path.join(features_dir, "y_train.npy"))
y_val = np.load(os.path.join(features_dir, "y_val.npy"))
y_test = np.load(os.path.join(features_dir, "y_test.npy"))

with open(os.path.join(features_dir, "features_meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

label_mapping = meta["label_mapping"]
id_to_label = {v: k for k, v in label_mapping.items()}
target_names = [id_to_label[i] for i in sorted(id_to_label.keys())]

phishing_label_id = label_mapping.get("phishing", None)

logger.info(f"X_train shape: {X_train.shape}")
logger.info(f"X_val shape: {X_val.shape}")
logger.info(f"X_test shape: {X_test.shape}")
logger.info(f"Labels: {label_mapping}")
logger.info(f"Phishing label id: {phishing_label_id}")

def eval_and_log_results(model_name, y_true, y_pred, split_name):
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )
    cm = confusion_matrix(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    logger.info(f"\n===== {model_name} - {split_name} - Classification Report =====\n{report}")
    logger.info(f"{model_name} - {split_name} - Macro F1: {macro_f1:.4f}")
    logger.info(f"{model_name} - {split_name} - Weighted F1: {weighted_f1:.4f}")
    logger.info(f"{model_name} - {split_name} - Confusion Matrix:\n{cm}")

    phishing_f1 = None
    phishing_precision = None
    phishing_recall = None

    if phishing_label_id is not None:
        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=sorted(id_to_label.keys())
        )
        idx = sorted(id_to_label.keys()).index(phishing_label_id)
        phishing_precision = precisions[idx]
        phishing_recall = recalls[idx]
        phishing_f1 = f1s[idx]
        logger.info(
            f"{model_name} - {split_name} - PHISHING metrics -> "
            f"Precision: {phishing_precision:.4f}, "
            f"Recall: {phishing_recall:.4f}, "
            f"F1: {phishing_f1:.4f}"
        )

    report_path = os.path.join(
        models_dir,
        f"{model_name.lower().replace(' ', '_')}_{split_name.lower()}_report.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\nSplit: {split_name}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(f"Macro F1: {macro_f1:.6f}\n")
        f.write(f"Weighted F1: {weighted_f1:.6f}\n")
        if phishing_f1 is not None:
            f.write(
                f"Phishing Precision: {phishing_precision:.6f}\n"
                f"Phishing Recall: {phishing_recall:.6f}\n"
                f"Phishing F1: {phishing_f1:.6f}\n"
            )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "phishing_f1": phishing_f1
    }

models = []

svc = LinearSVC(
    class_weight="balanced",
    random_state=random_state
)
models.append(("LinearSVC", svc))

logreg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced"
)
models.append(("LogisticRegression", logreg))

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=random_state,
    class_weight=None
)
models.append(("RandomForest", rf))

best_model_name = None
best_model = None
best_val_macro_f1 = -1.0
best_val_phishing_f1 = -1.0
results_summary = {}

for model_name, model in models:
    logger.info(f"\n\n############################")
    logger.info(f"Training model: {model_name}")
    logger.info(f"############################")

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_metrics = eval_and_log_results(model_name, y_val, y_val_pred, "VAL")

    results_summary[model_name] = {
        "val_macro_f1": val_metrics["macro_f1"],
        "val_phishing_f1": val_metrics["phishing_f1"]
    }

    if val_metrics["phishing_f1"] is not None:
        better = False
        if val_metrics["phishing_f1"] > best_val_phishing_f1 + 1e-6:
            better = True
        elif abs(val_metrics["phishing_f1"] - best_val_phishing_f1) < 1e-6 and \
                val_metrics["macro_f1"] > best_val_macro_f1 + 1e-6:
            better = True

        if better:
            best_val_phishing_f1 = val_metrics["phishing_f1"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_model_name = model_name
            best_model = model

logger.info("\n================ SUMMARY (Validation) ================")
for name, metrics in results_summary.items():
    logger.info(
        f"{name}: val_macro_f1={metrics['val_macro_f1']:.4f}, "
        f"val_phishing_f1={metrics['val_phishing_f1']:.4f}"
    )

if best_model is None:
    logger.error("No best model selected. Check training.")
else:
    logger.info(
        f"\nBest model (by phishing F1, then macro F1) on VAL: {best_model_name} "
        f"(phishing F1={best_val_phishing_f1:.4f}, macro F1={best_val_macro_f1:.4f})"
    )

    best_model_path = os.path.join(models_dir, f"best_model_{best_model_name}.joblib")
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved best model to: {best_model_path}")

    logger.info("\nEvaluating best model on TEST set")
    y_test_pred = best_model.predict(X_test)
    test_metrics = eval_and_log_results(best_model_name, y_test, y_test_pred, "TEST")

    summary_path = os.path.join(models_dir, "best_model_test_summary.json")
    summary_payload = {
        "best_model_name": best_model_name,
        "val_macro_f1": best_val_macro_f1,
        "val_phishing_f1": best_val_phishing_f1,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_phishing_f1": test_metrics["phishing_f1"]
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    logger.info("Training and evaluation complete.")
    logger.info(f"Test summary: {json.dumps(summary_payload, indent=2)}")
