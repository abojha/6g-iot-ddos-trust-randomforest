---

```markdown
# 🚀 Trust-Based DDoS Detection in 6G IoT Networks

This project presents a lightweight, explainable, and scalable machine learning framework for detecting Distributed Denial-of-Service (DDoS) attacks in 6G-enabled IoT networks. It uses a **Spatial-Temporal Trust Model** to evaluate the trustworthiness of IoT nodes based on their behavior and employs a **Random Forest Classifier** to distinguish between benign and malicious traffic.

---

## 📌 Project Overview

With the growth of 6G and IoT, traditional DDoS detection systems struggle to handle the heterogeneity and scale of device traffic. This framework solves that by:
- Calculating **spatial trust** (deviation from peer behavior)
- Calculating **temporal trust** (deviation from node's historical behavior)
- Aggregating trust into node- and flow-level scores
- Using these features to train a Random Forest model

---

## 🧠 Core Methodology

### Feature Engineering:
- `k1`: Activity (number of flows)
- `k2`: Traffic symmetry
- `k3`: Average packet size
- `k4`: Destination entropy

### Trust Calculation:
- **Spatial Trust** compares a node’s behavior to its cluster.
- **Temporal Trust** predicts behavior via linear regression over time.
- **Flow Trust** evaluates trust per individual packet flow.

### Model:
- Balanced the dataset using undersampling
- Trained a Random Forest classifier on extracted trust-based features

---

## 📊 Dataset

- **Bot-IoT Dataset** with 3.6M+ flows
- 21 unique IoT nodes
- 13 time slots
- Two-class labels: 0 (Benign), 1 (Attack)

---

## ✅ Results

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 99.98%    |
| Precision    | 99.98%    |
| Recall       | 99.98%    |
| ROC AUC      | 0.9999    |
| PR AUC       | 0.9999    |

Confusion matrix and evaluation metrics outperform traditional ML models like SVM and RNN, making the system ideal for real-time edge deployments.

---

## 🛠️ Installation & Usage

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ddos-trust-6g-iot.git
cd ddos-trust-6g-iot
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Place Dataset
Add your **preprocessed Bot-IoT CSV file** to the root directory as:
```
preprocessed_dataset.csv
```

### 4. Run feature extraction & training
```bash
python random_forest.py
```

### 5. View Results
All outputs including:
- `features.csv`
- `randomForestResults/`:
  - `roc_curve.png`
  - `pr_curve.png`
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `random_forest_log.txt`

---

## 📁 Repository Structure

```
├── t.py                    # Trust feature extraction pipeline
├── random_forest.py        # Main training and evaluation script
├── dataset_stats.py        # Dataset statistics generator
├── features.csv            # Saved feature data
├── randomForestResults/    # Logs and evaluation plots
├── README.md               # Project documentation
```

---

## 🧪 Example Visuals

| ROC Curve | PR Curve | Confusion Matrix |
|-----------|----------|------------------|
| ![ROC](randomForestResults/roc_curve.png) | ![PR](randomForestResults/pr_curve.png) | ![CM](randomForestResults/confusion_matrix.png) |

---

## 📚 References

- Yinglun Ma et al., “DDoS Detection for 6G IoT: Spatial-Temporal Trust Model and New Architecture,” *IEEE China Communications*, 2022.
- Bot-IoT Dataset: [https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)

---

## 👨‍💻 Author

**Abhay Ojha**  
M.Tech, Computer Science and Information Security  
National Institute of Technology, Warangal  
Supervisor: Prof. Rashmi Ranjan Rout

---

## 📬 Contact
[LinkedIn](https://www.linkedin.com/in/abhayojha0012) or email.
---
