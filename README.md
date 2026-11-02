# Student Lighthouse

Prediction tool for student dropout risk, built for the Machine Learning × Advanced Python final project (aivancity, PGE 2).

**Student Lighthouse** helps a university's student success service decide *who to call first*. Given a student's admission file and 1st-semester results, it predicts the risk of dropout and generates a readable "Student Card" with actionable signals.

## Source

Dataset: *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository, id 697.
License: CC BY 4.0.
Link: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

## Install

We recommend using a virtual environment.

python -m venv venv
.\venv\Scripts\Activate.ps1

""" 1-#On Terminal or Bash:"""

pip install -e .

""" 2-To use the Streamlit dashboard, also install the optional dependencies:"""

pip install -e ".[app]"

## Commands

""" 1-Train the model and save it:"""

lighthouse train

""" 2-Evaluate the saved model on the test set (prints metrics):"""

lighthouse evaluate

""" 3-Predict a single student from a JSON file:"""

lighthouse predict --input new_case.json

(Two example files are provided: new_case.json for a low-risk student and high_risk_case.json for a high-risk student).

""" 4-Launch the interactive dashboard:"""

streamlit run src/lighthouse/app.py

(Requires the [app] optional dependencies)

## Run the tests

pytest

## Project Structure

src/lighthouse/ : The core Python package.

data/raw/ : The raw dataset (CSV).

models/ : Where the trained model is saved.

tests/ : Unit tests (cleaning, split, no leakage, CLI, etc.).

DECISIONS.md : Records of all trade-offs made during the project.

src/lighthouse/app.py : Streamlit dashboard (Triage simulator, Model Lab, Methodology).

## Credits

Hugo KINGKEOMANIVONG & Pierre MATAR — aivancity, PGE2 EN.