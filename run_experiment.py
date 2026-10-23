"""Reproducible synthetic proof-of-concept for readiness classification.
IMPORTANT: the bundled data are synthetic, not observations from real enterprises.
Replace them with a licensed, documented real-world dataset before empirical claims.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
FEATURES = [
    "Data governance", "Analytics capability", "Leadership commitment",
    "Innovation investment", "Digital skills", "Process integration",
    "Technology infrastructure", "Change readiness", "Cybersecurity maturity",
    "Ecosystem collaboration"
]

def main():
    X, y = make_classification(
        n_samples=600, n_features=10, n_informative=6, n_redundant=2,
        n_classes=3, n_clusters_per_class=2, weights=[.34,.38,.28],
        class_sep=1.0, flip_y=.06, random_state=802
    )
    data = pd.DataFrame(X, columns=FEATURES)
    data["readiness_class"] = y
    data.to_csv(OUT / "synthetic_demo_dataset.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=802)
    models = {
        "Multinomial logistic regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=802))
        ]),
        "Random forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                class_weight="balanced", random_state=802, n_jobs=-1))
        ])
    }
    rows = []
    baseline = []
    for train, test in cv.split(X, y):
        pred = np.repeat(np.bincount(y[train]).argmax(), len(test))
        baseline.append((accuracy_score(y[test], pred),
                         f1_score(y[test], pred, average="macro"),
                         balanced_accuracy_score(y[test], pred)))
    rows.append(["Majority-class baseline", *np.mean(baseline, axis=0)])
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring={
            "accuracy": "accuracy", "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy"
        }, n_jobs=-1)
        rows.append([name, scores["test_accuracy"].mean(),
                     scores["test_macro_f1"].mean(), scores["test_balanced_accuracy"].mean()])
    result = pd.DataFrame(rows, columns=["Model", "Accuracy", "Macro-F1", "Balanced accuracy"])
    result.to_csv(OUT / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(result)); width = .24
    for j, metric in enumerate(["Accuracy", "Macro-F1", "Balanced accuracy"]):
        ax.bar(x + (j - 1) * width, result[metric], width, label=metric)
    ax.set_xticks(x); ax.set_xticklabels(result["Model"], rotation=12, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Mean 5-fold cross-validation score")
    ax.set_title("Synthetic proof-of-concept (not enterprise evidence)")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / "figure2_model_comparison.png", dpi=300)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
