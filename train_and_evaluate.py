"""
UNSW-NB15 Data Preprocessing and XGBoost Baseline Model Training
Clean pipeline for train/test preprocessing and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("=" * 70)
print("STEP 1: Loading Data")
print("=" * 70)

train_df = pd.read_excel(r"C:\Users\ldako\Downloads\UNSW_NB15_training-set(in).xlsx")
print(f"Train set loaded! Initial size: {train_df.shape}")

test_df = pd.read_csv(r"C:\Users\ldako\Downloads\UNSW_NB15_testing-set(in).csv")
print(f"Test set loaded! Initial size: {test_df.shape}")

# ============================================================================
# STEP 2: Drop Unnecessary Columns
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Dropping Unnecessary Columns")
print("=" * 70)

train_df = train_df.drop(['id', 'attack_cat'], axis=1)
test_df = test_df.drop(['id', 'attack_cat'], axis=1)

print(f"Train set after dropping columns: {train_df.shape}")
print(f"Test set after dropping columns: {test_df.shape}")

# ============================================================================
# STEP 3: Split Features and Target
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Splitting Features and Target")
print("=" * 70)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ============================================================================
# STEP 4: Encode Categorical Variables (Dummy Variables)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Encoding Categorical Variables")
print("=" * 70)

categorical_cols = ['proto', 'service', 'state']

X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

print(f"X_train shape after get_dummies: {X_train.shape}")
print(f"X_test shape after get_dummies: {X_test.shape}")

# Align X_test columns with X_train (fill missing columns with 0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"X_test shape after column alignment: {X_test.shape}")
print(f"Columns match: {X_train.shape[1] == X_test.shape[1]}")

# ============================================================================
# STEP 5: MinMax Normalization
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: MinMax Normalization")
print("=" * 70)

scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on training data and transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Clip test values to [0, 1] range (test set can have values outside train range)
X_test_scaled = np.clip(X_test_scaled, 0, 1)

# Convert back to DataFrames to preserve column names
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"X_train - Min: {X_train.min().min():.4f}, Max: {X_train.max().max():.4f}")
print(f"X_test - Min: {X_test.min().min():.4f}, Max: {X_test.max().max():.4f}")

# ============================================================================
# STEP 6: Train XGBoost Baseline Model
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Training XGBoost Baseline Model")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    objective='binary:logistic',
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42,
    verbosity=1
)

xgb_model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# STEP 7: Make Predictions and Evaluate
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Model Evaluation")
print("=" * 70)

# Predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print results
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

# ============================================================================
# STEP 8: ROC Curve Plot
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Plotting ROC Curve")
print("=" * 70)

plt.figure(figsize=(10, 7))
RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.title("ROC Curve - XGBoost Baseline Model", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
print("ROC curve saved as 'roc_curve.png'")
plt.show()

print("\n" + "=" * 70)
print("Pipeline completed successfully!")
print("=" * 70)
