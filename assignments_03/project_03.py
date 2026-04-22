import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import (
confusion_matrix,
accuracy_score,
precision_score,
recall_score,
f1_score,
classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore", category=RuntimeWarning)


COLUMN_NAMES = [
"word_freq_make", # 0 percent of words that are "make"
"word_freq_address", # 1
"word_freq_all", # 2
"word_freq_3d", # 3 almost never appears
"word_freq_our", # 4
"word_freq_over", # 5
"word_freq_remove", # 6 common in "remove me from this list"
"word_freq_internet", # 7
"word_freq_order", # 8
"word_freq_mail", # 9
"word_freq_receive", # 10
"word_freq_will", # 11
"word_freq_people", # 12
"word_freq_report", # 13
"word_freq_addresses", # 14
"word_freq_free", # 15 classic spam word
"word_freq_business", # 16
"word_freq_email", # 17
"word_freq_you", # 18
"word_freq_credit", # 19
"word_freq_your", # 20 often high in spam
"word_freq_font", # 21 HTML emails
"word_freq_000", # 22 "win $ x,000" style offers
"word_freq_money", # 23 money related
"word_freq_hp", # 24 HP specific
"word_freq_hpl", # 25
"word_freq_george", # 26 specific HP person
"word_freq_650", # 27 area code
"word_freq_lab", # 28
"word_freq_labs", # 29
"word_freq_telnet", # 30
"word_freq_857", # 31
"word_freq_data", # 32
"word_freq_415", # 33
"word_freq_85", # 34
"word_freq_technology", # 35
"word_freq_1999", # 36
"word_freq_parts", # 37
"word_freq_pm", # 38
"word_freq_direct", # 39
"word_freq_cs", # 40
"word_freq_meeting", # 41
"word_freq_original", # 42
"word_freq_project", # 43
"word_freq_re", # 44 reply threads
"word_freq_edu", # 45
"word_freq_table", # 46
"word_freq_conference", # 47
"char_freq_;", # 48 frequency of ';'
"char_freq_(", # 49 frequency of '('
"char_freq_[", # 50 frequency of '['
"char_freq_!", # 51 exclamation marks (often big)
"char_freq_$", # 52 dollar sign (money related)
"char_freq_#", # 53 hash character
"capital_run_length_average", # 54 average length of capital letter runs
"capital_run_length_longest", # 55 longest capital run
"capital_run_length_total", # 56 total number of capital letters
"spam_label" # 57 1 = spam, 0 = not spam
]

# TASK 1: LOAD AND EXPLORE

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())


plt.figure(figsize=(8, 5))
plt.boxplot(df.word_freq_free)
plt.title("word_freq_free by spam label")
plt.xlabel("spam_label (0=ham, 1=spam)")
plt.ylabel("word_freq_free")
plt.show()
plt.savefig("outputs/boxplot_word_freq__free.png")

plt.figure(figsize=(8, 5))
plt.boxplot(df["char_freq_!"])
plt.title("char_freq_! by spam label")
plt.xlabel("spam_label (0=ham, 1=spam)")
plt.ylabel("char_freq_!")
plt.show()
plt.savefig("outputs/boxplot_char_freq_!")

plt.figure(figsize=(8, 5))
plt.boxplot(df.capital_run_length_total)
plt.title("Capital_run_length_total by spam label")
plt.xlabel("spam_label (0=ham, 1=spam)")
plt.ylabel("capital_run_length_total")
plt.show()
plt.savefig("outputs/boxplot_capital_run_lenght_total.png")

# The results on the graphs are concetrated towards zero, because most email do not contain these words.


# TASK 2: PREPARE YOUR DATA

X = df.drop(columns="spam_label")
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print("Number of PCA components to reach 90% variance:", n_components_90)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(y=0.90, linestyle="--", label="90% variance")
plt.axvline(x=n_components_90, linestyle="--", label=f"{n_components_90} components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.title("PCA Cumulative Variance")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_cumulative_variance.png")
plt.show()
plt.close()

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]


# TASK 3: A CLASSIFIER COMPARISON

results = {}

# KNN (unscaled)

knn_unscaled = KNeighborsClassifier(n_neighbors=5)

knn_unscaled.fit(X_train, y_train)

knn_unscaled_pred = knn_unscaled.predict(X_test)

knn_unscaled_acc = accuracy_score(y_test, knn_unscaled_pred)

print("\n=== KNN (unscaled) ===")
print(f"Accuracy: {knn_unscaled_acc:.4f}")
print(classification_report(y_test, knn_unscaled_pred))

results["KNN unscaled"] = {
    "accuracy": knn_unscaled_acc,
    "predctions": knn_unscaled_pred,
    "model": knn_unscaled
}

# KNN (scaled)
knn_scaled = KNeighborsClassifier(n_neighbors=5)

knn_scaled.fit(X_train_scaled, y_train)
knn_scaled_pred = knn_scaled.predict(X_test_scaled)

knn_scaled_acc = accuracy_score(y_test, knn_scaled_pred)

print("=== KNN (scaled) ===")
print(f"Accuracy: {knn_scaled_acc:.4f}")
print(classification_report(y_test, knn_scaled_pred))

results["KNN scaled"] = {
    "accuracy": knn_scaled_acc,
    "predictions": knn_scaled_pred,
    "model": knn_scaled
}

# KNN (PCA)
knn_pca = KNeighborsClassifier(n_neighbors=5)

knn_pca.fit(X_train_pca, y_train)
knn_pca_pred = knn_pca.predict(X_test_pca)

knn_pca_acc = accuracy_score(y_test, knn_pca_pred)

print("\n=== KNN (PCA) ===")
print(f"Accuracy: {knn_pca_acc:.4f}")
print(classification_report(y_test, knn_pca_pred))

results["KNN PCA"] = {
    "accuracy": knn_pca_acc,
    "preditions": knn_pca_pred,
    "model": knn_pca
}

# Decision Tree depth comparison

depths = [3, 5, 10, None]

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)

    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    test_pred = tree.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"=== Decision Tree max_depth={depth} ===")
    print("Train Accuracy:", train_acc)
    print("Test Accurcy:",  test_acc)


decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

decision_tree.fit(X_train, y_train)
decision_tree_pred = decision_tree.predict(X_test)

decision_tree_acc = accuracy_score(y_test, decision_tree_pred)

print("=== Decision Tree (max_depth=5) ===")
print("Accuracy:", decision_tree_acc)
print(classification_report(y_test, decision_tree_pred))

results["Decision Tree"] = {
    "accuracy": decision_tree_acc,
    "predictions": decision_tree_pred,
    "model": decision_tree
}


# Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("=== Random Forest ===")
print("Accuracy:", rf_acc)
print(classification_report(y_test, rf_pred))

results["Random Forest"] = {
    "accuracy": rf_acc,
    "predictions": rf_pred,
    "model": rf
}


# Logistic Regression scaled

log_reg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")

log_reg_scaled.fit(X_train_scaled, y_train)
log_reg_scaled_pred = log_reg_scaled.predict(X_test_scaled)

log_reg_scaled_acc = accuracy_score(y_test, log_reg_scaled_pred)

print("=== Logistic Regression (scaled) ===")
print("Accuracy:", log_reg_scaled_acc)
print(classification_report(y_test, log_reg_scaled_pred))

results["Logistic Regression scaled"] = {
    "accuracy": log_reg_scaled_acc,
    "predictions": log_reg_scaled_pred,
    "model": log_reg_scaled
}


# Logistic Regression PCA

log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")

log_reg_pca.fit(X_train_pca, y_train)
log_reg_pca_pred = log_reg_pca.predict(X_test_pca)

log_reg_pca_acc = accuracy_score(y_test, log_reg_pca_pred)

print("\n=== Logistic Regression (PCA) ===")
print(f"Accuracy: {log_reg_pca_acc:.4f}")
print(classification_report(y_test, log_reg_pca_pred))

results["Logistic Regression PCA"] = {
    "accuracy": log_reg_pca_acc,
    "predictions": log_reg_pca_pred,
    "model": log_reg_pca
}


# Summary

print("=== MODEL ACCURACY SUMMARY ===")
for name, info in results.items():
    print(f"{name}: {info['accuracy']:.4f}")


# Confusion Matrix

best_model_name = max(results, key=lambda name: results[name]["accuracy"])
best_model_info = results[best_model_name]

print("\nBest model:", best_model_name)
print(f"Best accuracy: {best_model_info['accuracy']:.4f}")

best_predictions = best_model_info["predictions"]

cm = confusion_matrix(y_test, best_predictions)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

if fp > fn:
    print("This model makes more false positives than false negatives.")
elif fn > fp:
    print("This model makes more false negatives than false positives.")
else:
    print("This model makes the same number of false positives and false negatives.")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.title(f"Confusion Matrix: {best_model_name}")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.show()


# TASK 4: CROSS - VALIDATION

cv_results = {}


# KNN unscaled

knn_unscaled_cv = KNeighborsClassifier(n_neighbors=5)
knn_unscaled_scores = cross_val_score(knn_unscaled_cv, X_train, y_train, cv=5)

knn_unscaled_mean = knn_unscaled_scores.mean()
knn_unscaled_std = knn_unscaled_scores.std()

print("\n=== KNN (unscaled) Cross-Validation ===")
print("Fold scores:", knn_unscaled_scores)
print(f"Mean accuracy: {knn_unscaled_mean:.4f}")
print(f"Std accuracy:  {knn_unscaled_std:.4f}")

cv_results["KNN unscaled"] = {
    "mean_accuracy": knn_unscaled_mean,
    "std_accuracy": knn_unscaled_std
}


# KNN scaled

knn_scaled_cv = KNeighborsClassifier(n_neighbors=5)
knn_scaled_scores = cross_val_score(knn_scaled_cv, X_train_scaled, y_train, cv=5)

knn_scaled_mean = knn_scaled_scores.mean()
knn_scaled_std = knn_scaled_scores.std()

print("\n=== KNN (scaled) Cross-Validation ===")
print("Fold scores:", knn_scaled_scores)
print(f"Mean accuracy: {knn_scaled_mean:.4f}")
print(f"Std accuracy:  {knn_scaled_std:.4f}")

cv_results["KNN scaled"] = {
    "mean_accuracy": knn_scaled_mean,
    "std_accuracy": knn_scaled_std
}


# KNN PCA

knn_pca_cv = KNeighborsClassifier(n_neighbors=5)
knn_pca_scores = cross_val_score(knn_pca_cv, X_train_pca, y_train, cv=5)

knn_pca_mean = knn_pca_scores.mean()
knn_pca_std = knn_pca_scores.std()

print("\n=== KNN (PCA) Cross-Validation ===")
print("Fold scores:", knn_pca_scores)
print(f"Mean accuracy: {knn_pca_mean:.4f}")
print(f"Std accuracy:  {knn_pca_std:.4f}")

cv_results["KNN PCA"] = {
    "mean_accuracy": knn_pca_mean,
    "std_accuracy": knn_pca_std
}


# Decision Tree

decision_tree_cv = DecisionTreeClassifier(max_depth=5, random_state=42)
decision_tree_scores = cross_val_score(decision_tree_cv, X_train, y_train, cv=5)

decision_tree_mean = decision_tree_scores.mean()
decision_tree_std = decision_tree_scores.std()

print("\n=== Decision Tree Cross-Validation ===")
print("Fold scores:", decision_tree_scores)
print(f"Mean accuracy: {decision_tree_mean:.4f}")
print(f"Std accuracy:  {decision_tree_std:.4f}")

cv_results["Decision Tree"] = {
    "mean_accuracy": decision_tree_mean,
    "std_accuracy": decision_tree_std
}


# Random Forest

random_forest_cv = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_scores = cross_val_score(random_forest_cv, X_train, y_train, cv=5)

random_forest_mean = random_forest_scores.mean()
random_forest_std = random_forest_scores.std()

print("\n=== Random Forest Cross-Validation ===")
print("Fold scores:", random_forest_scores)
print(f"Mean accuracy: {random_forest_mean:.4f}")
print(f"Std accuracy:  {random_forest_std:.4f}")

cv_results["Random Forest"] = {
    "mean_accuracy": random_forest_mean,
    "std_accuracy": random_forest_std
}


# Logistic Regression scaled

log_reg_scaled_cv = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_reg_scaled_scores = cross_val_score(log_reg_scaled_cv, X_train_scaled, y_train, cv=5)

log_reg_scaled_mean = log_reg_scaled_scores.mean()
log_reg_scaled_std = log_reg_scaled_scores.std()

print("\n=== Logistic Regression (scaled) Cross-Validation ===")
print("Fold scores:", log_reg_scaled_scores)
print(f"Mean accuracy: {log_reg_scaled_mean:.4f}")
print(f"Std accuracy:  {log_reg_scaled_std:.4f}")

cv_results["Logistic Regression scaled"] = {
    "mean_accuracy": log_reg_scaled_mean,
    "std_accuracy": log_reg_scaled_std
}


# Logistic Regression PCA

log_reg_pca_cv = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_reg_pca_scores = cross_val_score(log_reg_pca_cv, X_train_pca, y_train, cv=5)

log_reg_pca_mean = log_reg_pca_scores.mean()
log_reg_pca_std = log_reg_pca_scores.std()

print("\n=== Logistic Regression (PCA) Cross-Validation ===")
print("Fold scores:", log_reg_pca_scores)
print(f"Mean acuracy: {log_reg_pca_mean:.4f}")
print(f"Std accuracy:  {log_reg_pca_std:.4f}")

cv_results["Logistic Regression PCA"] = {
    "mean_accuracy": log_reg_pca_mean,
    "std_accuracy": log_reg_pca_std
}


# Summary

print("=== CROSS-VALIDATION SUMMARY ===")
for name, info in cv_results.items():
    print(f"{name}: mean={info['mean_accuracy']:.4f}, std={info['std_accuracy']:.4f}")

most_accurate_model = max(cv_results, key=lambda name: cv_results[name]["mean_accuracy"])
most_stable_model = min(cv_results, key=lambda name: cv_results[name]["std_accuracy"])

print("Most accurat model:", most_accurate_model)
print(f"Mean accuracy: {cv_results[most_accurate_model]['mean_accuracy']:.4f}")

print("Most stable model:", most_stable_model)
print(f"Std accuracy: {cv_results[most_stable_model]['std_accuracy']:.4f}")



# The cross validation gives average results in multiple folds of the training data, that's why is
# consider better than the train/test split.

# The most accurate model is the Random forrest, it has the highest mean.  And the most stable model is
# the Logistic Regression PCA because it has the lowest standard deviation.


# TASK 5: BUILDING PREDICTION PIPELINES


# RANDOM FORES AND LOGISTIC REGRESSION SCALE HAVE BETTER MEANS

# Tree-based pipeline

tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

tree_pipeline.fit(X_train, y_train)
tree_pipeline_pred = tree_pipeline.predict(X_test)

tree_pipeline_acc = accuracy_score(y_test, tree_pipeline_pred)

print("\n=== Tree-Based Pipeline (Random Forest) ===")
print(f"Accuracy: {tree_pipeline_acc:.4f}")
print(classification_report(y_test, tree_pipeline_pred))

print("Earlier manual Random Forest accuracy:", results["Random Forest"]["accuracy"])
print("Pipeline Random Forest accuracy:", tree_pipeline_acc)


# Non-tree pipeline

non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

non_tree_pipeline.fit(X_train, y_train)
non_tree_pipeline_pred = non_tree_pipeline.predict(X_test)

non_tree_pipeline_acc = accuracy_score(y_test, non_tree_pipeline_pred)

print("\n=== Non-Tree Pipeline (Logistic Regression scaled) ===")
print(f"Accuracy: {non_tree_pipeline_acc:.4f}")
print(classification_report(y_test, non_tree_pipeline_pred))

print("Earlier manual Logistic Regression scaled accuracy:", results["Logistic Regression scaled"]["accuracy"])
print("Pipeline Logistic Regression accuracy:", non_tree_pipeline_acc)

# A pipeline combines the preprocessing steps and the model into one object.
# This makes things easier because I don’t have to manually scale or transform
# the data every time, and it reduces the chance of making mistakes like
# applying steps in the wrong order.

# The tree-based pipeline is simpler because tree models don’t need scaling or PCA.
# They work by splitting on feature values, so they are not affected by the size
# of the numbers like KNN or Logistic Regression.

# The non-tree pipeline includes scaling because Logistic Regression depends on
# feature scale. 

# The pipelines do not have the same structure.
# The tree pipeline only has the model, while the non-tree pipeline includes
# preprocessing steps like scaling and PCA if needed.

# The main benefit of using pipelines is that everything is done in one place.
# It is especially useful if I want to reuse the model later or deploy it, since 
# the same steps will always be applied to new data.