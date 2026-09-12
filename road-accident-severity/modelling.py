
# 1. IMPORT LIBRARIES

# Standard / utility libraries
import os #for operating system related operations
import json #for json files
from pathlib import Path

# Saving / loading models
import joblib 

# Data and math
import numpy as np # numerical operations 
import pandas as pd # data manipulation in an array/table format

# Plotting
import matplotlib.pyplot as plt  #both are used for plotting
import seaborn as sns # for enhanced visualizations

# Scikit-learn: ML utilities libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# special tools for imbalanced classes
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Gradient boosting powerful ML algorithms libraries
import xgboost as xgb
import lightgbm as lgb

# 2. PATHS AND GLOBAL SETTINGS

# Root folder for this project
PROJECT_ROOT = Path("/Users/sanyawadhawan/Desktop/road-accident-severity")

# Path to the cleaned dataset file (CSV)
DATA_PATH = Path("/Users/sanyawadhawan/Desktop/data_labeled_new.csv")

# to save trained models and reports/plots
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

#If the models or reports folders don’t exist, please create them. 
# If they already exist, don’t crash, just continue.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Fix randomness so results are repeatable( used for reshuffling data)
RANDOM_STATE = 42

# 3. LOAD DATA AND DEFINE TARGET

print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset shape (rows, columns):", df.shape)
print("First 5 rows:")
print(df.head(5))

# this is the column we want to predict -(Y value)
TARGET = "CRASH_TYPE"

#dropping crashid column as its not useful for prediction- will result in overfitting
#DROP_COLS = ["Crash ID","No. killed","No. seriously injured","No. moderately injured","No. minor-other injured"]

# Safety check: make sure target column exists
assert TARGET in df.columns, f"Target column '{TARGET}' not found in data!"

# X = input features (independent variables), y = target labels (dependent variable)
X = df.drop(columns=[TARGET]) # drop target and unwanted columns(crash id)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df[TARGET])  # ensure labels are integers and setting Y=target
joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

print("\nFeature columns (X):", list(X.columns))
print(pd.Series(y).value_counts(normalize=True).sort_index())

# For now, we treat all columns as numeric features
numeric_features = X.columns.tolist()
print("\nNumber of numeric features:", len(numeric_features))

 # 4. TRAIN / TEST SPLIT

# Split the data so we can evaluate/test on unseen examples
# stratify=y keeps class proportions similar in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,        # 20% test, 80% train
    stratify=y,    #when we split the data, we want to maintain the same class distribution of the target variable in both train and test sets
    #eg: if class 1 is 30% of data, it should be 30% in both train and test
    random_state=RANDOM_STATE,
)

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)
print("Class distribution in full data:")
print(pd.Series(y).value_counts(normalize=True).sort_index())#proportion of each class in the target variable across the entire dataset

# 5. PREPROCESSING PIPELINE

# For numeric features: fill missing values with median, then standardize
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),  # during .fit() An imputer fills in missing values (NaN) in your data with the median value of that feature
        ("scaler", StandardScaler()),                  # scale the features (mean 0, std 1). during .transform()
    ]
)
#we use median to avoid outliers

# ColumnTransformer lets us apply different preprocessing to different columns
# Here, all features are numeric, so we just use numeric_transformer on all
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ],
    remainder="drop",  # drop any columns not listed (there shouldn't be any) basically any non numeric columns
)


# 6. DEFINE MODELS (WITH IMBALANCE HANDLING)

# 6.1 Logistic Regression with class weights
#linear model for binary and multiclass classification
#outputs probabilities for each class
pipe_logreg = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                multi_class="auto",
                solver="saga", #this is good for large datasets and supports regularization and optimization.
                class_weight="balanced",  # handle class imbalance
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

# 6.2 Extra Trees (it is a descision tree ensemble) with class weights. 
pipe_extratrees = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            ExtraTreesClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,  # use all CPU cores
            ),
        ),
    ]
)

# 6.3 XGBoost classifier (no explicit class weights here)
#uses gradient boosting on decision trees
pipe_xgb = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            xgb.XGBClassifier(
                objective="binary:logistic",   # 🔑 for 0/1 target
                eval_metric="logloss",         # binary log loss
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

# 6.4 LightGBM classifier
pipe_lgbm = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            lgb.LGBMClassifier(
                objective="binary",            # 🔑 binary classification
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)


# 6.5 Logistic Regression + SMOTE (oversample minority classes)
smote = SMOTE(random_state=RANDOM_STATE)

pipe_logreg_smote = ImbPipeline(
    steps=[
        ("preprocess", preprocess),
        ("smote", smote),  # apply SMOTE on training folds only
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                multi_class="auto",
                solver="saga",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# 7. CROSS-VALIDATION SETUP
#when skikit-learn trains the model it needs to compute metrics such as F1 score and balanced accuracy to evaluate performance.
# We will evaluate models using these metrics:
scoring = {
    "macro_f1": make_scorer(f1_score, average="macro"),   #average of F1 scores for each class(F1=2*(precision*recall)/(precision+recall))
    "bal_acc": make_scorer(balanced_accuracy_score), #average of recall obtained on each class
}

# 5-fold stratified cross-validation
#4 folds for training, 1 fold for validation, repeat 5 times
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

# FUNCTION TO RUN CROSS-VALIDATION AND COLLECT RESULTS IN A SUMMARY DICTIONARY
def evaluate_with_cv(model, name):   #model is the pipeline, name is string name of model

    print(f"\nRunning CV for model: {name}")
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,  # use all CPU cores for speed
        return_estimator=False,
    )

    summary = {
        "model": name,
        "macro_f1_mean": float(np.mean(cv_results["test_macro_f1"])),
        "macro_f1_std": float(np.std(cv_results["test_macro_f1"])),
        "bal_acc_mean": float(np.mean(cv_results["test_bal_acc"])),
        "bal_acc_std": float(np.std(cv_results["test_bal_acc"])),
    }

    print("CV results for", name, ":", summary)
    return summary


# List of (model, name) pairs to evaluate
models_to_try = [
    (pipe_logreg, "LogisticRegression"),
    (pipe_extratrees, "ExtraTrees"),
    (pipe_xgb, "XGBoost"),
    (pipe_lgbm, "LightGBM"),
    (pipe_logreg_smote, "LogReg+SMOTE"),
]

# Run CV for each model and collect results
all_cv_results = []
for mdl, name in models_to_try:
    try:
        result = evaluate_with_cv(mdl, name)
        all_cv_results.append(result)
    except Exception as e:
        print(f"Error while evaluating {name}: {e}")
        all_cv_results.append({"model": name, "error": str(e)})

cv_results_df = pd.DataFrame(all_cv_results) #convert list of dicts to dataframe for better visualization
print("\nCross-validation summary:")
print(cv_results_df)

# ===== Visualize CV Results =====
cv_plot = cv_results_df.copy()
cv_plot = cv_plot[["model", "macro_f1_mean", "bal_acc_mean"]].set_index("model")

cv_plot.plot(kind="bar", figsize=(10,6), rot=45, edgecolor="black")
plt.title("Cross-Validation Metrics Comparison")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



# 8. SELECT BEST MODEL AND TRAIN ON FULL TRAINING DATA

# Filter out models that had errors
valid_results = [r for r in all_cv_results if "error" not in r]

if len(valid_results) > 0:
    cv_results_valid_df = pd.DataFrame(valid_results)
    # Choose model with highest mean macro F1
    best_row = cv_results_valid_df.sort_values(
        "macro_f1_mean", ascending=False
    ).iloc[0]
    best_model_name = best_row["model"]
else:
    # If everything failed somehow, default to ExtraTrees
    best_model_name = "ExtraTrees"

print("\nBest model based on CV macro F1:", best_model_name)

# Map from name to actual pipeline object
# to actually train and predict, you need the actual pipeline object, e.g. pipe_xgb.
#So model_map is just a dictionary that connects the string name to the actual model:

model_map = {
    "LogisticRegression": pipe_logreg,
    "ExtraTrees": pipe_extratrees,
    "XGBoost": pipe_xgb,
    "LightGBM": pipe_lgbm,
    "LogReg+SMOTE": pipe_logreg_smote,
}

best_model = model_map[best_model_name]

# Fit the selected model on ALL training data
print("\nFitting the best model on full training data...")
best_model.fit(X_train, y_train)  #standardize and find mean on full training data

# ===== Feature Importance =====
if best_model_name in ["ExtraTrees", "XGBoost", "LightGBM"]:
    model_feat = best_model.named_steps['clf']
    importances = model_feat.feature_importances_
    features = X.columns
    feat_df = pd.DataFrame({"feature": features, "importance": importances})
    feat_df = feat_df.sort_values(by="importance", ascending=False).head(20)  # top 20

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=feat_df, palette="viridis", ax=ax)
    ax.set_title(f"Top 20 Feature Importances - {best_model_name}")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / f"feature_importance_{best_model_name}.png", dpi=150)
    plt.show()

# 9. EVALUATE ON TEST SET

# Predictions on test data
y_pred = best_model.predict(X_test)

# Try to get predicted probabilities (not all models may support this)
try:
    y_proba = best_model.predict_proba(X_test)
except Exception:
    y_proba = None

# Compute test metrics
test_bal_acc = balanced_accuracy_score(y_test, y_pred)
test_macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n=== Test set performance ===")
print("Best model:", best_model_name)
print("Test Balanced Accuracy:", test_bal_acc)
print("Test Macro F1:", test_macro_f1)
print("\nClassification report:\n")
print(classification_report(y_test, y_pred,target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_labels = le.classes_ #get original class labels from label encoder
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
ax.set_title(f"Confusion Matrix - {best_model_name}")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
plt.show()



# 10. SAVE ARTIFACTS (MODEL, METRICS, PLOTS, PROBAS)


# Save the trained best model
model_path = MODELS_DIR / "best_pipeline.joblib"
joblib.dump(best_model, model_path)
print("\nSaved best model to:", model_path)

# Save cross-validation results (all models)
cv_results_path = REPORTS_DIR / "cv_results.csv"
cv_results_df.to_csv(cv_results_path, index=False)
print("Saved CV results to:", cv_results_path)

# Save test metrics in JSON format
metrics = {
    "best_model": best_model_name,
    "test_balanced_accuracy": test_bal_acc,
    "test_macro_f1": test_macro_f1,
    "classes": le.classes_.tolist(),
}
metrics_path = REPORTS_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved test metrics to:", metrics_path)


#Save test probabilities (if available), useful for risk maps
if y_proba is not None:
    proba_df = pd.DataFrame(y_proba, index=X_test.index)
    proba_path = REPORTS_DIR / "test_proba.parquet"
    proba_df.to_parquet(proba_path)
    print("Saved test probabilities to:", proba_path)

print("\nDone!")

