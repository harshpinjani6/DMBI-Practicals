"""DMBI Experiment 10: Telecom Customer Churn Business Intelligence.
Run: pip install pandas numpy scikit-learn matplotlib seaborn
     python customer_churn_bi.py
Place Telco-Customer-Churn.csv (or WA_Fn-UseC_-Telco-Customer-Churn.csv)
in the same folder, or pass its path as the first command-line argument.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, silhouette_score)

sns.set_theme(style="whitegrid")
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
    (p for p in [Path('Telco-Customer-Churn.csv'),
                 Path('WA_Fn-UseC_-Telco-Customer-Churn.csv')]
     if p.exists()), None)
if DATA is None or not DATA.exists():
    raise FileNotFoundError('Place the Telco churn CSV next to this script or pass its path.')
df = pd.read_csv(DATA)
required = {'customerID', 'TotalCharges', 'Churn', 'tenure', 'MonthlyCharges', 'Contract'}
if not required.issubset(df.columns):
    raise ValueError(f'Dataset missing columns: {required - set(df.columns)}')
print('Raw dataset:', df.shape)
print(df[['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn']].head())

# Data preprocessing
missing = pd.to_numeric(df['TotalCharges'], errors='coerce').isna().sum()
print('Missing TotalCharges:', missing)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
clean = df.drop(columns=['customerID']).copy()
clean['Churn'] = clean['Churn'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(clean.drop(columns='Churn'), drop_first=True)
y = clean['Churn']
print('Encoded feature matrix:', X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
print('Training:', X_train.shape, 'Testing:', X_test.shape)

# Supervised classification
models = {
    'Decision Tree (CART)': DecisionTreeClassifier(max_depth=5, criterion='gini',
                                                   class_weight='balanced', random_state=42),
    'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=200, max_depth=8,
        min_samples_leaf=10, class_weight='balanced', random_state=42)
}
rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plt.figure(figsize=(7, 5))
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({'Algorithm': name, 'Accuracy': accuracy_score(y_test, pred),
                 'Precision': precision_score(y_test, pred),
                 'Recall': recall_score(y_test, pred),
                 'F1-Score': f1_score(y_test, pred),
                 'ROC-AUC': roc_auc_score(y_test, prob)})
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set(title=name, xlabel='Predicted', ylabel='Actual')
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.figure(2)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, prob):.3f})')
fig.tight_layout()
fig.savefig('exp10_confusion_matrices.png', dpi=150)
plt.figure(2)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('exp10_roc_curves.png', dpi=150)
print('\nModel evaluation:')
print(pd.DataFrame(rows).round(4).to_string(index=False))

# Churn by contract
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Customer Churn by Contract Type')
plt.tight_layout()
plt.savefig('exp10_churn_by_contract.png', dpi=150)

# Explainable feature importance
rf = models['Random Forest (Ensemble)']
importance = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10).sort_values()
plt.figure(figsize=(9, 5))
importance.plot(kind='barh')
plt.title('Top 10 Churn Drivers (Random Forest)')
plt.tight_layout()
plt.savefig('exp10_feature_importance.png', dpi=150)

# K-Means segmentation of customers who churned
churners = df[df['Churn'] == 'Yes'].copy()
cluster_features = ['tenure', 'MonthlyCharges']
cluster_scaler = StandardScaler()
cluster_x = cluster_scaler.fit_transform(churners[cluster_features])
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
churners['Cluster'] = kmeans.fit_predict(cluster_x)
score = silhouette_score(cluster_x, churners['Cluster'])
print('\nK-Means silhouette score:', round(score, 4))
print('Cluster profiles:')
print(churners.groupby('Cluster')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean().round(2))
plt.figure(figsize=(9, 5))
sns.scatterplot(data=churners, x='tenure', y='MonthlyCharges', hue='Cluster', palette='deep', s=20)
plt.title('K-Means Segmentation of Churned Customers')
plt.tight_layout()
plt.savefig('exp10_kmeans_clusters.png', dpi=150)
print('\nSaved 5 visualization files.')
