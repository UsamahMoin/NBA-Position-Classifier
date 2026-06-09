# NBA Position Classifier

An interactive, research-backed visualization of a linear support vector
machine that predicts 2021 NBA player positions from per-game statistics.

**Live site:** https://usamahmoin.github.io/NBA-Position-Classifier/

## What the project shows

- A browser reproduction of the trained linear SVM's one-vs-one vote.
- Live controls for all 12 features retained by recursive feature elimination.
- Held-out accuracy, balanced accuracy, a confusion matrix, and per-class
  precision/recall/F1 scores.
- Original class counts compared with the SMOTE-balanced training set.
- Feature importance and position-level statistical profiles.

## Method

The dataset is cleaned by keeping the combined `TOT` record for players who
changed teams. A stratified 75/25 split is made before preprocessing. Mean
imputation, standardization, SMOTE, recursive feature elimination, and
hyperparameter tuning are fit on training data only. The untouched test split
is used once for the reported final evaluation.

This order matters: applying scaling or SMOTE before the split would allow
information from the test set to influence training and produce an optimistic
evaluation.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 Project2.py
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Rebuild the browser model

```bash
python3 tools/train_model.py
```

This regenerates `model-artifact.json` from `nba2021.csv`.
