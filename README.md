# AI for Business Analytics — Teaching Case + Streamlit Decision App

This repository contains:
- A Harvard-style teaching case (see `CASE_HARVARD_STYLE.md`)
- A Streamlit application that demonstrates the end-to-end decision pipeline taught in the course:
  - Problem formulation (state/action/goal/cost/constraints)
  - Search (BFS/DFS/UCS/A*), heuristic design, complexity trade-offs
  - Metaheuristics (hill climbing, simulated annealing, genetic algorithm – lightweight demos)
  - Supervised ML workflow (features → train → evaluate → tune)
  - Metrics for classification/regression
  - Linear models, trees, ensembles
  - Feature engineering, leakage checks
  - Interpretability (feature importance, permutation importance; optional SHAP if installed)
  - Bias/imbalance handling and drift monitoring
  - Simple neural network demo (MLP via scikit-learn)

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Data
Synthetic datasets in `data/`:
- `churn.csv` (classification)
- `demand.csv` (regression)
- `fraud.csv` (imbalanced classification)

## Structure
- `app.py` Streamlit UI entrypoint
- `src/` core logic
- `CASE_HARVARD_STYLE.md` teaching case (student version)
- `TEACHING_NOTE.md` instructor note

## License
Educational use.
