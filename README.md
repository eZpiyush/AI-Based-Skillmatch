# SkillMatch AI — AICTE Machine Learning Project

## Overview
SkillMatch AI is a small, runnable career exploration prototype. Users select skills and receive ranked role suggestions based on text similarity.

## Machine learning / NLP method
- **TF-IDF** represents skill text using weighted word features.
- **Cosine similarity** compares the user's skill text with each role's requirements.
- Roles are sorted by similarity score.
- Scores are similarity measures, **not** hiring probabilities.

## Requirements
- Python 3.10 or newer recommended
- VS Code and Python extension

## Run in VS Code (Windows)
1. Extract the ZIP and open the `SkillMatch_AI_AICTE_Project` folder in VS Code.
2. Open Terminal → New Terminal.
3. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or install packages with `python -m pip`.
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   python -m streamlit run app.py
   ```
6. A local browser tab should open. Select skills and click **Find suitable roles**.

## Files
- `app.py` — Streamlit interface and matching logic
- `data/job_roles.csv` — illustrative role/skill dataset
- `requirements.txt` — Python packages
- `docs/` — submission documents

## Limitations and responsible use
The included dataset is small and illustrative, not a representative labour-market dataset. Matching quality depends on vocabulary overlap. The prototype does not predict salary, hiring outcomes, or employment probability. Do not use it to make high-stakes decisions without validation and human review.

## Suggested demo
Try `Python`, `SQL`, and `Pandas`; then try `HTML`, `CSS`, and `JavaScript`; compare the ranked roles and download the result CSV.
