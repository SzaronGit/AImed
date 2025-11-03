from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv("task_data.csv", decimal=",") #float error


X = data[[
    "Heart width", "Lung width", "CTR - Cardiothoracic Ratio",
    "xx", "yy", "xy", "normalized_diff", "Inscribed circle radius",
    "Polygon Area Ratio", "Heart perimeter", "Heart area ", "Lung area"
]]

y = data["Cardiomegaly"]
y = data["Cardiomegaly"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

#-------------------------------------------------------------------------------------------------------------
print("K-Nearest Neighbors (KNN) Classifier")

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors = 3,
        weights='distance',
        metric='manhattan'
    ))
])

pipe_knn.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_knn, X_train, y_train), 2)

print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")


param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],
    "model__weights": ["uniform", "distance"],
    "model__metric": ["minkowski", "manhattan", "euclidean", "chebyshev"],
}

rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=100,
    random_state=None
)

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid_search = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy (averaged CV): {grid_search.best_score_:.4f}\n")


param_grid = {
    "model__n_neighbors": [6, 7, 8],
    "model__weights": ["uniform"],
    "model__metric": ["manhattan"],
}

rskf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=100,
    random_state=None
)

pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])

grid_search = GridSearchCV(
    estimator=pipe_knn,
    param_grid=param_grid,
    scoring="accuracy",
    cv=rskf,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy (averaged CV): {grid_search.best_score_:.4f}\n")


pipe_knn = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors = 7,
        weights='uniform',
        metric='manhattan'
    ))
])

pipe_knn.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_knn, X_train, y_train), 2)

print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#-------------------------------------------------------------------------------
print("Decision Tree")

clf_tree = DecisionTreeClassifier(
    max_depth=7,
    criterion='log_loss',
    min_samples_split=7,
    min_samples_leaf=5,
    class_weight=None
)

clf_tree.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_tree, X_train, y_train), 2)

print(f"Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#-----------------------------------------------------------------------------------------------
print("Support vector Machine (SVM)")

pipe_svc = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        C=3,
        gamma="scale",
        class_weight=None
    ))
])

pipe_svc.fit(X_train, y_train)
cv_score = np.round(cross_val_score(pipe_svc, X_train, y_train), 2)

print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {cv_score.mean():.3f}")
print(f"Standard deviation of CV score: {cv_score.std():.3f}\n")

#-------------------------------------------------------------------------------------
print("Logistic Regression")

pipe_log = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=1,
        penalty="l1",
        solver="liblinear",
        max_iter=1000,
        class_weight=None
    ))
])

pipe_log.fit(X_train, y_train)

cv_score = np.round(cross_val_score(pipe_log, X_train, y_train), 2)

print("Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {cv_score.mean():.3f}")
print(f"Standard deviation of CV score: {cv_score.std():.3f}\n")

#--------------------------------------------------------------------------------------------------
print("Random Forest Classifier")

clf_rf = RandomForestClassifier(
    max_depth=6,
    min_samples_split=6,
    n_estimators=125,
    min_samples_leaf=2,
    max_features='sqrt'
)

clf_rf.fit(X_train, y_train)

cv_score = np.round(cross_val_score(clf_rf, X_train, y_train), 2)

print(f"Scores of training data cross-validation (each fold):")
list(map(print, cv_score))
print(f"\nCross-validation mean score: {np.mean(cv_score):.3f}")
print(f"Standard deviation of CV score: {np.std(cv_score):.3f}\n")

#-------------------------------------------------------------------------------------------------
print("Running the Model on Test Dataset")

y_pred_knn  = pipe_knn.predict(X_test)
y_pred_svc  = pipe_svc.predict(X_test)
y_pred_log  = pipe_log.predict(X_test)
y_pred_tree = clf_tree.predict(X_test)
y_pred_rf   = clf_rf.predict(X_test)

acc_knn  = accuracy_score(y_test, y_pred_knn)
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_svc  = accuracy_score(y_test, y_pred_svc)
acc_log  = accuracy_score(y_test, y_pred_log)
acc_rf   = accuracy_score(y_test, y_pred_rf)

prec_knn  = precision_score(y_test, y_pred_knn)
prec_tree = precision_score(y_test, y_pred_tree)
prec_svc  = precision_score(y_test, y_pred_svc)
prec_log  = precision_score(y_test, y_pred_log)
prec_rf   = precision_score(y_test, y_pred_rf)

rec_knn  = recall_score(y_test, y_pred_knn)
rec_tree = recall_score(y_test, y_pred_tree)
rec_svc  = recall_score(y_test, y_pred_svc)
rec_log  = recall_score(y_test, y_pred_log)
rec_rf   = recall_score(y_test, y_pred_rf)

f1_knn  = f1_score(y_test, y_pred_knn)
f1_tree = f1_score(y_test, y_pred_tree)
f1_svc  = f1_score(y_test, y_pred_svc)
f1_log  = f1_score(y_test, y_pred_log)
f1_rf   = f1_score(y_test, y_pred_rf)

print(f"Accuracy on test set:")
print(f"- Accuracy of KNN Classifier model on test dataset:              {acc_knn:.4f}")
print(f"- Accuracy of Decision Tree model on test dataset:    {acc_tree:.4f}")
print(f"- Accuracy of SVC model on test dataset:              {acc_svc:.4f}")
print(f"- Accuracy of Logistic Regression model on test dataset:    {acc_log:.4f}")
print(f"- Accuracy of Random Forest model on test dataset:    {acc_rf:.4f}\n")

print(f"Precision on test set:")
print(f"- Precision of KNN Classifier model on test dataset:              {prec_knn:.4f}")
print(f"- Precision of Decision Tree model on test dataset:    {prec_tree:.4f}")
print(f"- Precision of SVC model on test dataset:              {prec_svc:.4f}")
print(f"- Precision of Logistic Regression model on test dataset:    {prec_log:.4f}")
print(f"- Precision of Random Forest model on test dataset:    {prec_rf:.4f}\n")

print(f"Recall on test set:")
print(f"- Recall of KNN Classifier model on test dataset:              {rec_knn:.4f}")
print(f"- Recall of Decision Tree model on test dataset:    {rec_tree:.4f}")
print(f"- Recall of SVC model on test dataset:              {rec_svc:.4f}")
print(f"- Recall of Logistic Regression model on test dataset:    {rec_log:.4f}")
print(f"- Recall of Random Forest model on test dataset:    {rec_rf:.4f}\n")

print(f"F1-Score on test set:")
print(f"- F1-Score of KNN Classifier model on test dataset:              {f1_knn:.4f}")
print(f"- F1-Score of Decision Tree model on test dataset:    {f1_tree:.4f}")
print(f"- F1-Score of SVC model on test dataset:              {f1_svc:.4f}")
print(f"- F1-Score of Logistic Regression model on test dataset:    {f1_log:.4f}")
print(f"- F1-Score of Random Forest model on test dataset:    {f1_rf:.4f}\n")
