# Student Lighthouse

**Dropout-risk triage for student success teams.**  
A Python package, CLI and Streamlit dashboard that turn a student’s admission file and first-semester results into a clear risk score and an actionable **Student Card**.

Built for the **Machine Learning × Advanced Python** final project at **aivancity, PGE 2**.

---

## Overview

Student Lighthouse helps a university’s student success service answer a concrete question:

> **Who should we call first?**

Given roughly 4,400 students and a strict capacity limit of 200 personal follow-ups per semester, the tool estimates the risk of dropout and produces a readable summary with the main signals and a recommended action. It is designed as a **decision-support tool**, not a black-box score: the goal is to help a human team prioritise outreach before the second semester begins.

---

## What makes this project interesting

- **End-to-end ML pipeline** — data cleaning, feature selection, training, evaluation and single-student prediction.
- **Package-first design** — the model is exposed through a reusable Python package and a CLI, not only a notebook.
- **Leakage-aware modelling** — the decision date is explicit; second-semester features are strictly removed because they are not available when the team needs to act.
- **Readable output** — predictions are formatted as a Student Card with risk level, main signals and suggested action.
- **Tested** — unit tests cover cleaning, splitting, leakage checks and CLI behaviour.
- **Documented trade-offs** — every important decision is recorded in `DECISIONS.md`.
- **Interactive dashboard** — a Streamlit app provides a Triage simulator, a Model Lab and a Methodology view.
- **Explicit model selection and hyperparameter tuning** — three models compared under a leakage-safe cross-validation protocol, with the final choice justified in `DECISIONS.md`.

---

## Modelling approach and hyperparameter tuning

The project goes beyond a single default model. The modelling pipeline is explicit, reproducible and designed to avoid optimistic results.

### Preprocessing pipeline

- Categorical and numerical features are handled separately in a scikit-learn `ColumnTransformer`.
- Numerical columns are standardized (`StandardScaler`) and categorical codes are One-Hot Encoded (`OneHotEncoder(handle_unknown="ignore")`).
- Preprocessing is fitted only on training folds, then applied to validation and test data, to prevent data leakage.
- The target is binarised for triage: `Dropout = 1`, `Enrolled + Graduate = 0`.

### Models compared

The following models were compared under the same preprocessing and validation protocol:
- **Logistic Regression** (interpretable baseline)
- **Linear SVM** (margin-based classifier)
- **k-Nearest Neighbors** (instance-based classifier)

### Hyperparameter tuning

Hyperparameters were tuned with **`GridSearchCV`** using **5-fold Stratified Cross-Validation**. The optimization metric was **Average Precision (PR-AUC)**, chosen because the dataset is imbalanced (32% dropout rate) and the capacity constraint requires ranking the most at-risk students accurately.

The search space included:
- **Logistic Regression:** `C` ∈ {0.1, 1.0, 10.0} — Best: `C = 1.0` (`class_weight='balanced'`, `max_iter=1000`)
- **Linear SVM:** `C` ∈ {0.1, 1.0, 10.0} — Best: `C = 0.1` (`class_weight='balanced'`)
- **k-NN:** `n_neighbors` ∈ {3, 5, 7, 11} — Best: `n_neighbors = 11`

The final model was selected on cross-validation performance, not on the test set. The test set was used once, at the end, for the final evaluation.

### Handling class imbalance

The dataset is imbalanced, so accuracy alone is not a reliable metric. The project uses:
- **`class_weight='balanced'`** for Logistic Regression and Linear SVM.
- **Average Precision (PR-AUC)** as the primary metric, which focuses on the ranking of the most at-risk students.
- **A decision threshold of 0.9927**, chosen on out-of-fold predictions to select exactly the top 200 students, respecting the real-world capacity constraint.

### Final retained model

The retained model is **Logistic Regression** (`C = 1.0`, `class_weight='balanced'`), selected for its performance (CV score of 0.8556) and full interpretability. It provides coefficients that translate directly into the readable "Student Card" used by the advisor.

This model is saved in `models/` and reused by the CLI (`lighthouse predict`) and the Streamlit dashboard, ensuring that the notebook, the package and the app rely on the same trained artefact.

---

## Tech stack

- **Python**
- **scikit-learn / pandas** for modelling and data processing
- **pytest** for tests
- **Streamlit** for the optional dashboard
- **CLI entry point** via `lighthouse`

---

## Quickstart

We recommend using a virtual environment.

```bash
git clone https://github.com/Keogo811/student-lighthouse
cd student-lighthouse

python -m venv venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

To use the Streamlit dashboard, also install the optional dependencies:

```bash
pip install -e ".[app]"
```

---

## Commands

| Command | What it does |
|---|---|
| `lighthouse train` | Train the model and save it. |
| `lighthouse evaluate` | Evaluate the saved model on the held-out test set and print metrics. |
| `lighthouse predict --input new_case.json` | Predict a single student from a JSON file. |
| `streamlit run src/lighthouse/app.py` | Launch the interactive dashboard. |

Two example files are provided:

- `new_case.json` — a low-risk student.
- `high_risk_case.json` — a high-risk student.

Example:

```bash
lighthouse predict --input high_risk_case.json
```

---

## Run the tests

```bash
pytest
```

The test suite covers cleaning, splitting, leakage prevention, CLI behaviour and other core invariants (such as verifying the scaler is fitted only on the training set).

---

## Project structure

```text
src/lighthouse/          Core Python package
data/raw/                Raw dataset (CSV)
models/                  Saved trained model
tests/                   Unit tests
DECISIONS.md             Trade-offs and design decisions
src/lighthouse/app.py    Streamlit dashboard
```

The dashboard includes:

- **Triage simulator** — explore how a student’s profile affects the risk level.
- **Model Lab** — inspect model behaviour and evaluation.
- **Methodology** — understand the modelling choices and limitations.

---

## Design decisions that matter

### 1. Explicit decision date and leakage prevention

The model is built around a clear decision point: what is known **at the end of the first semester**. Second-semester features are strictly removed because they would not be available when the student success team needs to act. Using future data is temporal leakage. This choice is documented, tested and reflected in the feature set, at the cost of lower lab performance.

### 2. Binary triage target

The original dataset contains three outcomes: `Dropout`, `Enrolled` and `Graduate`. For a triage tool, the useful question is binary: **is this student at risk of dropping out, or not?** We grouped `Enrolled` and `Graduate` as "not dropout" because an enrolled student has not dropped out yet, and we want to detect actual departure, not academic delay. The mapping is documented in `DECISIONS.md`.

### 3. Evaluation that respects the test set

Model selection and threshold choices are made without touching the final test set. The test set is used once, at the end, to report an honest evaluation. This is especially important because the dataset is imbalanced: accuracy alone would be misleading.

### 4. CLI-first reproducibility

The package is designed so that a clean environment can run:

```bash
pip install -e .
pytest
lighthouse train
lighthouse evaluate
lighthouse predict --input new_case.json
```

This makes the project reproducible and easy to inspect.

### 5. Human-readable output and interpretability

The prediction is not only a number. It is formatted as a **Student Card** with a risk level, the main contributing signals and a recommended action, so that a non-technical user can understand and act on it. The Logistic Regression coefficients reveal that the most influential features are:
1. **Curricular units 1st sem (approved)** — Negative sign. More approved credits significantly reduce risk.
2. **Curricular units 1st sem (grade)** — Negative sign. Higher grades mean lower risk.
3. **Tuition fees up to date** — Negative sign. Unpaid fees correlate with higher risk.
4. **Age at enrollment** — Positive sign. Older students have a slightly higher risk.

These signals are exactly what a human advisor would look at. The model is not a black box.

---

## Dataset

**Predict Students’ Dropout and Academic Success**  
UCI Machine Learning Repository, id 697 (4,424 rows, 36 features).  
License: **CC BY 4.0**.

Link: [https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

---

## Results

The final model is evaluated with:

```bash
lighthouse evaluate
```

Metrics are printed to stdout and discussed in the project report. The held-out test set is used only once, after model selection, to keep the evaluation honest.

| Metric | Value |
|---|---|
| Average Precision (PR-AUC) | 0.8727 |
| ROC-AUC | 0.9120 |
| Precision @ Top 50 (scaled capacity) | 0.9800 (98%) |
| Selected threshold | 0.9927 |

At the chosen threshold (0.9927), the model achieves a Precision@K of 98% on the test set (57 true positives out of 58 predicted positives). This means that when the model says "call this student", it is almost always right. The confusion matrix is `[[600, 1], [227, 57]]`. Threshold justification and the cost of false negatives vs. false positives are documented in `DECISIONS.md`.

---

## Limitations and Risks

- **The "Enrolled" ambiguity:** Grouping "Enrolled" with "Graduate" means the model might under-estimate the risk for students who are currently struggling but have not officially dropped out yet.
- **The cost of removing S2 data:** Dropping second-semester features is the honest choice for real-world deployment, but it lowers the model's performance compared to a leaky model.
- **No qualitative data:** The dataset only contains administrative and academic metrics. It cannot capture a student's mental health, motivation, or personal struggles.
- **Stigmatization:** The "High Risk" label must be presented as a conversation starter, not a verdict. The human advisor makes the final decision.
- **The 200-slot cutoff:** A student ranked 201st is ignored, even if their risk score is 0.9926 (just below the threshold). This is a harsh reality of limited resources.
- **Fairness:** The model could inadvertently learn biases from historical data based on age, gender, or nationality. A proper fairness audit would be mandatory before production use.

---

## Credits

**Hugo KINGKEOMANIVONG & Pierre MATAR**  
aivancity, PGE2 EN.
