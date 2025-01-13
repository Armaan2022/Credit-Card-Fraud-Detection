import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data: Training set (70%), Temp set (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training Set Fraud Ratio: {y_train.mean() * 100:.2f}%")
print(f"Validation Set Fraud Ratio: {y_val.mean() * 100:.2f}%")
print(f"Test Set Fraud Ratio: {y_test.mean() * 100:.2f}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Baseline Model
logistic_baseline = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000, class_weight='balanced')
logistic_baseline.fit(X_train_scaled, y_train)

y_val_pred = logistic_baseline.predict(X_val_scaled)
y_val_proba = logistic_baseline.predict_proba(X_val_scaled)[:, 1]

precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')

print("Baseline Model Results (Logistic Regression Without Regularization):")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


# Logistic Regression
logistic = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
logistic_params = {'C': uniform(0.01, 10)}
logistic_search = RandomizedSearchCV(logistic, logistic_params, scoring='f1', cv=5, n_iter=10, random_state=42, n_jobs=-1)
logistic_search.fit(X_train_scaled, y_train)

best_logistic = logistic_search.best_estimator_
print(f"Best Logistic Regression Parameters: {logistic_search.best_params_}")


# k-NN 
knn = KNeighborsClassifier()
knn_params = {
    'n_neighbors': randint(3, 15), 
    'metric': ['minkowski']
}
knn_search = RandomizedSearchCV(knn, knn_params, scoring='f1', cv=5, n_iter=10, random_state=42, n_jobs=-1)
knn_search.fit(X_train_scaled, y_train)

best_knn = knn_search.best_estimator_
print(f"Best k-NN Parameters: {knn_search.best_params_}")


# Random Forest
random_forest = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_params = {
    'n_estimators': randint(50, 500), 
    'max_depth': [10, 20, None], 
}
rf_search = RandomizedSearchCV(random_forest, rf_params, scoring='f1', cv=5, n_iter=10, random_state=42, n_jobs=-1)
rf_search.fit(X_train_scaled, y_train)

best_rf = rf_search.best_estimator_
print(f"Best Random Forest Parameters: {rf_search.best_params_}")



# Results

models = {
    "Baseline Logistic Regression": logistic_baseline,
    "Tuned Logistic Regression": best_logistic,
    "k-NN": best_knn,
    "Random Forest": best_rf
}

for model_name, model in models.items():
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
    
    print(f"{model_name} Results:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_test_pred))


# Plots

# Precision-Recall Curve

plt.figure(figsize=(10, 8))

for model_name, model in models.items():
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    avg_precision = average_precision_score(y_test, y_test_proba)
    plt.plot(recall, precision, label=f"{model_name} (AP = {avg_precision:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for Models")
plt.legend(loc="lower left")
plt.grid(alpha=0.5)
plt.savefig('Precision-Recall'+'.jpg')


# F1-Score Comparison Curve

model_names = ["Baseline Logistic Regression", "Tuned Logistic Regression", "k-NN", "Random Forest"]
f1_scores = []

for model in models.values():
    y_test_pred = model.predict(X_test_scaled)
    f1 = precision_recall_fscore_support(y_test, y_test_pred, average="binary")[2]
    f1_scores.append(f1)

plt.figure(figsize=(8, 6))
colors = ['gray', 'orange', 'blue', 'green']
plt.bar(model_names, f1_scores, color=colors)
plt.title("F1-Score Comparison")
plt.ylabel("F1-Score")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.5)
plt.savefig('F1-score'+'.jpg')