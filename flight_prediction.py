import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from Time_Series import X_test

train_path = Path("train.csv")
test_path = Path("test.csv")

if not train_path.exists():
    train_path = Path("/mnt/data/train.csv")

if not test_path.exists():
    test_path = Path("/mnt/data/test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

target = "ARR_DEL15"

#Remove rows where the target is missing
train = train.dropna(subset=[target])
test = test.dropna(subset=[target])

#Convert target to integer
train[target] = train[target].astype(int)
test[target] = test[target].astype(int)

def add_features(df):
    df["ROUTE"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)
    df["CRS_DEP_TIME_HOUR"] = df["CRS_DEP_TIME"] // 100
    df["CRS_DEP_MINUTE"] = df["CRS_DEP_TIME"] % 100
    df["CRS_ARR_TIME_HOUR"] = df["CRS_ARR_TIME"] // 100
    df["CRS_ARR_MINUTE"] = df["CRS_ARR_TIME"] % 100

    if "DAY_OF_WEEK" in df.columns:
        df["IS_WEEKEND"] = (df["DAY_OF_WEEK"].isin([6, 7])).astype(int)

    if "hourlyStationPressure" in df.columns:
        df["hourlyStationPressure"] = df["hourlyStationPressure"].replace(0, np.nan)

    return df

train = add_features(train)
test = add_features(test)

#Split features X and target Y first to prevent data leakage during preprocessing
X_train = train.drop(columns=[target])
y_train = train[target]
X_test = test.drop(columns=[target])
y_test = test[target]

categorical_features = [
    "OP_CARRIER", "ORIGIN", "DEST", "ROUTE"
]

categorical_features = [col for col in categorical_features if col in train.columns]

numerical_features = [
    col for col in train.columns if col not in categorical_features
]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = XGBClassifier(
    n_estimators = 300,
    max_depth = 5,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 42,
    objective = "binary:logistic",
    eval_metric = "logloss",
    n_jobs = -1
)
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

print("Training model...")
full_pipeline.fit(X_train, y_train)

print("Generating predictions...")
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")