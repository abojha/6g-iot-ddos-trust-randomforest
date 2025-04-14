# A Spatial-Temporal Trust-Based Random Forest Framework for DDoS Attack Detection in 6G IoT

This repository contains the implementation of a trust-based DDoS detection framework for IoT networks in 6G environments. The project computes trust values (spatial, temporal, flow-level) and classifies attack traffic using a Random Forest classifier.

---

## 📂 Project Structure

- `t.py` — Computes trust features from IoT traffic data.
- `random_forest.py` — Trains and evaluates the Random Forest model.
- `preprocessed_dataset.csv` — Preprocessed Bot-IoT dataset (not pushed to GitHub).
- `randomForestResults/` — Contains plots: ROC, PR, confusion matrix, etc.
- `report/` — Final LaTeX project report, figures, bibliography, and appendices.

---

## ⚙️ Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, tqdm, joblib

---

## 🚀 How to Run

```bash
# Step 1: Extract features (if not already done)
python random_forest.py

# Step 2: Model, plots, and logs will be saved to randomForestResults/
