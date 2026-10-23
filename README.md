# AI Decision Support for Digital Transformation Readiness

A compact, reproducible **synthetic proof-of-concept** accompanying the draft research paper.

## Important scientific limitation
The included observations are generated with `sklearn.datasets.make_classification`. They are not collected from enterprises and cannot support claims about real-world digital transformation readiness. Use a documented, licensed enterprise dataset and re-run the pipeline before making empirical claims or submitting an empirical manuscript.

## Run
```bash
python -m pip install -r requirements.txt
python run_experiment.py
```

The script writes a synthetic dataset and cross-validation comparison table to `results/`, and a programmatic model-comparison figure to `figures/`. The seed is fixed at 802.

## Minimal structure
- `run_experiment.py` — dataset generation, cross-validation and figure output
- `requirements.txt` — dependencies
- `README.md` — instructions and limitations
- `paper_draft.docx` — manuscript draft
- `figures/` — programmatically generated figures
- `results/` — generated outputs (synthetic only)

## Reproducibility notes
Stratified 5-fold cross-validation is used. All preprocessing that learns from data is placed inside scikit-learn pipelines. The synthetic benchmark is for validating code flow and figure/table generation only; it is not external validation.
