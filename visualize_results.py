import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.getcwd()
vis_dir = os.path.join(base_dir, "outputs", "visualizations")
os.makedirs(vis_dir, exist_ok=True)

models = ["LinearSVC", "LogisticRegression", "RandomForest"]

accuracy = np.array([0.9821, 0.9717, 0.9797])
precision = np.array([0.9787, 0.9568, 0.9804])
recall = np.array([0.9693, 0.9667, 0.9560])
macro_f1 = np.array([0.9739, 0.9615, 0.9677])
phishing_f1 = np.array([0.9396, 0.9107, 0.9314])

metrics = {
    "Accuracy": accuracy,
    "Macro Precision": precision,
    "Macro Recall": recall,
    "Macro F1": macro_f1,
}

x = np.arange(len(models))
bar_width = 0.18

plt.figure(figsize=(10, 6))
for i, (metric_name, values) in enumerate(metrics.items()):
    plt.bar(x + i * bar_width, values, width=bar_width, label=metric_name)

plt.xticks(x + bar_width * (len(metrics) - 1) / 2, models)
plt.ylim(0.85, 1.0)
plt.ylabel("Score")
plt.title("Model Performance ")
plt.legend(loc="lower right")

for i, (metric_name, values) in enumerate(metrics.items()):
    for j, v in enumerate(values):
        plt.text(x[j] + i * bar_width, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

perf_path = os.path.join(vis_dir, "models_performance.png")
plt.tight_layout()
plt.savefig(perf_path, dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
plt.bar(models, phishing_f1)
plt.ylim(0.85, 1.0)
plt.ylabel("F1-Score")
plt.title("Phishing Class F1-Score")

for i, v in enumerate(phishing_f1):
    plt.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

phish_path = os.path.join(vis_dir, "phishing_f1_comparison.png")
plt.tight_layout()
plt.savefig(phish_path, dpi=300)
plt.close()

print("Saved:")
print(perf_path)
print(phish_path)
