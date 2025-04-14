import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from trust_pipeline import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils import resample

# --- CONFIG ---
DATA_PATH = "preprocessed_dataset.csv"
FEATURE_PATH = "features.csv"
OUTPUT_DIR = "randomForestResults"
MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_model.pkl")
LOG_PATH = os.path.join(OUTPUT_DIR, "random_forest_log.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- STEP 1: Load or extract features ---
if not os.path.exists(FEATURE_PATH):
    print("Extracting features...")
    df_raw = pd.read_csv(DATA_PATH)
    features = extract_features(df_raw, t_slot=3600)
    features.to_csv(FEATURE_PATH, index=False)
    print(f"Extracted and saved {features.shape[0]} rows of features.")
else:
    print("Loading precomputed features...")
    features = pd.read_csv(FEATURE_PATH)
    print(f"Loaded {features.shape[0]} rows of features.")

# --- STEP 2: Node-based split (to prevent leakage) ---
nodes = features['saddr'].unique()
train_nodes, test_nodes = train_test_split(nodes, test_size=0.3, random_state=42)
train_mask = features['saddr'].isin(train_nodes)
test_mask = features['saddr'].isin(test_nodes)

X = features[['k1', 'k2', 'k3', 'k4', 'node_trust', 'TF']]
y = features['attack'].astype(int)
X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

# --- STEP 3: Undersample to balance classes ---
train_df = X_train.copy()
train_df['attack'] = y_train
benign = train_df[train_df['attack'] == 0]
attack = train_df[train_df['attack'] == 1]
# Balance by undersampling the majority class
if len(benign) > len(attack):
    benign_downsampled = resample(benign, replace=False, n_samples=len(attack), random_state=42)
    balanced_train = pd.concat([benign_downsampled, attack])
else:
    attack_downsampled = resample(attack, replace=False, n_samples=len(benign), random_state=42)
    balanced_train = pd.concat([benign, attack_downsampled])

X_train_balanced = balanced_train.drop('attack', axis=1)
y_train_balanced = balanced_train['attack']

# --- STEP 4: Train Random Forest ---
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
clf.fit(X_train_balanced, y_train_balanced)
joblib.dump(clf, MODEL_PATH)
print("Model trained and saved.")

# --- STEP 5: Predict and evaluate ---
y_proba = np.clip(clf.predict_proba(X_test)[:, 1], 0.01, 0.99)
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_idx = np.nanargmax(f1_scores)
best_thresh = thresholds[best_idx]
y_pred = (y_proba >= best_thresh).astype(int)

report = classification_report(y_test, y_pred, digits=4)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

# --- Cross-validation check ---
cv_scores = cross_val_score(clf, X_train_balanced, y_train_balanced, cv=3, scoring='f1')
print("Cross-validated F1 scores:", cv_scores)

# --- STEP 6: Log Results ---
with open(LOG_PATH, 'w') as f:
    f.write("==== Random Forest DDoS Detection ====" + "\n\n")
    f.write(f"Best threshold (F1): {best_thresh:.4f}\n")
    f.write(report + "\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"PR AUC: {pr_auc:.4f}\n")
    f.write(f"Cross-validated F1 scores: {cv_scores.tolist()}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

# --- STEP 7: Plots ---
plt.figure()
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
plt.close()

plt.figure()
plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.4f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'))
plt.close()

plt.figure()
plt.plot(thresholds, f1_scores[:-1])
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 vs Threshold")
plt.savefig(os.path.join(OUTPUT_DIR, 'f1_vs_threshold.png'))
plt.close()

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.xticks([0,1], ['Benign','Attack'])
plt.yticks([0,1], ['Benign','Attack'])
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

plt.figure()
fi = pd.Series(clf.feature_importances_, index=X.columns)
fi.plot(kind='bar')
plt.title('Feature Importance')
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
plt.close()

print("All plots, logs, and model saved.")
