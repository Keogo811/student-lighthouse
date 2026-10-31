# AI Usage Journal

This file records how generative AI tools were used during the project, as required by the course guidelines.

## Tool Used
- **DeepSeek** (used as an interactive coding assistant and tutor).

## How AI was used
- **Project Scaffolding:** Generating the initial boilerplate for `pyproject.toml` and the `src/` directory structure.
- **Scikit-learn Pipelines:** Drafting the template for the `ColumnTransformer` and `Pipeline` classes to ensure best practices (no data leakage).
- **CLI Implementation:** Writing the `argparse` skeleton for the `train`, `evaluate`, and `predict` commands.
- **Debugging:** Interpreting Python tracebacks and environment errors (e.g., handling the semicolon separator in the raw CSV, and fixing the `pyproject.toml` build configuration).
- **Documentation:** Drafting the structure of the `README.md` and `DECISIONS.md` templates.

## What we wrote, verified, and changed ourselves
- **The ML Logic:** We made the core decisions ourselves. We explicitly chose to remove the Semester 2 columns to prevent temporal leakage, we selected Average Precision (PR-AUC) to handle the class imbalance, and we calculated the custom decision threshold (0.9927) to match the 200-student capacity constraint.
- **Code Review:** The code was read, tested, reviewed and adjusted to our precise vision. For example, we modified the initial `predict.py` logic to prioritize numerical features so that the "Student Card" signals would be human-readable for the advisor and not an unreadable value.
- **Testing:** We wrote the logic for the unit tests and verified that all 12 tests pass on a fresh environment (testing for no leakage, correct shapes, and probability ranges).
- **Documentation:** The "Design Decisions" file reflects our own reasoning and trade-offs, and the report was written by us based on our own evaluation results and adapted in conscious to it.