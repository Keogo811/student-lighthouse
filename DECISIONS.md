# Design Decisions

This file records the trade-offs made during the project. Each choice lists the alternative we discarded and the reason why.

## 1. Target Mapping (Enrolled)
- **Choice:** `Graduate` and `Enrolled` become `0` (no dropout). `Dropout` becomes `1`.
- **Alternative:** Classifying `Enrolled` as dropout.
- **Reason:** An enrolled student has not dropped out yet. We predict actual departure, not academic delay.

## 2. Data Leakage (Semester 2)
- **Choice:** Drop all columns from the 2nd semester.
- **Alternative:** Keep them for a higher model score.
- **Reason:** At the decision date (end of S1), S2 data does not exist. Keeping them is temporal leakage: the model would be useless in production.

## 3. Metric
- **Choice:** Average Precision (PR-AUC) and Precision@200.
- **Alternative:** Accuracy or F1-score.
- **Reason:** The dataset is imbalanced (~32% dropout). Accuracy is misleading (a dummy model gets 68%). We need a metric that evaluates ranking, because the service has a limited capacity of 200 slots.

## 4. Model Selection
- **Choice:** Logistic Regression.
- **Alternative:** Linear SVM (0.8555 vs 0.8556).
- **Reason:** The scores are statistically identical. Logistic Regression gives readable coefficients. We can explain *why* a student is flagged to the advisor (Student Card).

## 5. Decision Threshold
- **Choice:** A dynamic threshold (0.9927) calculated to select exactly the top 200 students.
- **Alternative:** Default threshold of 0.5.
- **Reason:** 0.5 is arbitrary. Our threshold is based on real-world capacity (200 appointments per semester), making the tool directly actionable.

## 6. Streamlit Dashboard
- **Choice:** Build an interactive Streamlit interface for the defense and user demonstration.
- **Alternative:** A simple HTML/CSS page.
- **Reason:** Streamlit is 100% Python and so evaluated on this language, requires zero front-end code, and allows dynamic interactions (like the capacity slider) that directly illustrate our decision threshold concept to the jury.