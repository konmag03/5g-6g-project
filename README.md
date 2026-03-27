Explainable Classification of 5G Network Accesses (Normal vs Attack)

1. Project Overview
   This project implements binary classification for detecting malicious network accesses in a 5G environment using the UNSW-NB15 dataset.

The prediction task is:
0 → Normal (benign)
1 → Attack (malicious)

The pipeline includes:
-Data preprocessing for tabular machine learning
-Training and tuning of a Machine Learning model (XGBoost)
-Training and tuning of a Deep Learning model (MLP)
-Model evaluation using standard classification metrics
-ROC curve analysis and model comparison

2. Dataset
   UNSW-NB15 Dataset
   https://research.unsw.edu.au/projects/unsw-nb15-dataset

The dataset contains labeled network traffic features suitable for supervised learning.

Modeling choices:
Removed:
id (non-informative identifier)
attack_cat (not needed for binary classification)
Target variable:
label (0 = normal, 1 = attack)

3. Pipeline
   3.1 Preprocessing
   The following steps are applied:

-Load train/test datasets
-Remove non-relevant features (id, attack_cat)
-Split into features and target
-One-hot encoding for categorical features:
proto, service, state
-Alignment of train/test features after encoding
-Min-Max scaling to range [0, 1]

3.2 Machine Learning Model - XGBoost
Model: XGBClassifier
Objective: binary:logistic
Hyperparameter tuning using Optuna
Evaluation with 10-fold Stratified Cross Validation

Tuned parameters include:
-learning_rate
-max_depth
-gamma
-n_estimators

3.3 Deep Learning Model - MLP
Model: MLPClassifier (Multilayer Perceptron)
Fully connected neural network for tabular data
Hyperparameter tuning using Optuna
Evaluation with 10-fold Stratified Cross Validation

Tuned parameters include:
-Number of hidden layers
-Neurons per layer
-Activation function (relu, tanh)
-Learning rate
-L2 regularization (alpha)

3.4 Evaluation
Models are evaluated on the test set using:
-Accuracy
-Precision
-Recall
-F1-score
-ROC AUC

Additionally:
ROC curve for XGBoost
Comparative ROC curve (XGBoost vs MLP)

4. Results Summary
   XGBoost achieves high accuracy and stable performance
   MLP performs well but is more sensitive to hyperparameters

Overall:
XGBoost provides better separation between normal and attack traffic
Results are consistent with literature on tabular data modeling

5. Repository Structure
   project/
   │
   ├── preprocessing_train_and_evaluate.ipynb
   ├── dataset/
   │ ├── UNSW_NB15_training-set(in).csv
   │ └── UNSW_NB15_testing-set(in).csv
   │
   ├── roc_curve_tuned_xgboost.png
   └── README.md

6. Installation
   -Requirements
   -Python 3.10+
   -Required libraries:
   -pandas
   -numpy
   -matplotlib
   -scikit-learn
   -xgboost
   -optuna
   -jupyter
   -Install dependencies
   -pip install pandas numpy matplotlib scikit-learn xgboost optuna -jupyter

7. Running the Project
   Open the notebook:
   preprocessing_train_and_evaluate.ipynb
   Run all cells sequentially
   Important note:

The notebook may contain absolute file paths.
Update them to relative paths:

train_df = pd.read_csv('dataset/UNSW_NB15_training-set(in).csv')
test_df = pd.read_csv('dataset/UNSW_NB15_testing-set(in).csv')

8. Interpretation of Metrics
   Recall → ability to detect actual attacks
   Precision → reliability of attack predictions
   F1-score → balance between precision and recall
   ROC AUC → overall discrimination ability

In cybersecurity applications:

High recall is critical to avoid missing attacks, while maintaining acceptable precision.

9. Notes & Future Work
   The following extensions are planned for full compliance with the project requirements:

SHAP explainability (TreeSHAP & DeepSHAP)
Feature importance visualizations (beeswarm, waterfall, etc.)
Top-k feature selection experiment
LightSHAP comparison (performance & speed)

10. References
    -UNSW-NB15 Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
    -Optuna: https://optuna.org/
    -SHAP: https://shap.readthedocs.io/
    -Tabular ML discussion: https://mindfulmodeler.substack.com/p/tabularml-is-about-to-get-weird
    -Research paper: https://arxiv.org/abs/2207.08815
