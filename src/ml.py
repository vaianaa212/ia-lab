from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

@dataclass
class TrainReport:
    task: str
    model_name: str
    metrics: Dict[str, float]
    notes: str

def split_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre

def train_classifier(df: pd.DataFrame, target: str, model_key: str, test_size: float = 0.25, random_state: int = 42) -> Tuple[Pipeline, TrainReport, Dict[str, np.ndarray]]:
    X,y = split_xy(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
    pre = build_preprocess(X)

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "gb": GradientBoostingClassifier(random_state=random_state),
        "mlp": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=800, random_state=random_state)
    }
    if model_key not in models:
        raise ValueError(f"Unknown model_key: {model_key}")

    clf = Pipeline(steps=[("pre", pre), ("model", models[model_key])])
    clf.fit(Xtr, ytr)
    proba = None
    if hasattr(clf.named_steps["model"], "predict_proba"):
        proba = clf.predict_proba(Xte)[:,1]
    pred = clf.predict(Xte)

    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0))
    }
    if proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(yte, proba))
        except Exception:
            pass

    notes = "Pipeline: OneHotEncoder + StandardScaler + model. Metrics computed on test split."
    rep = TrainReport(task="classification", model_name=model_key, metrics=metrics, notes=notes)
    extra = {"y_true": yte.to_numpy(), "y_pred": pred, "proba": (proba if proba is not None else np.array([]))}
    return clf, rep, extra

def train_regressor(df: pd.DataFrame, target: str, model_key: str, test_size: float = 0.25, random_state: int = 42) -> Tuple[Pipeline, TrainReport, Dict[str, np.ndarray]]:
    X,y = split_xy(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=test_size,random_state=random_state)
    pre = build_preprocess(X)

    models = {
        "linreg": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
        "tree": DecisionTreeRegressor(max_depth=6, random_state=random_state),
        "rf": RandomForestRegressor(n_estimators=200, random_state=random_state),
        "gb": GradientBoostingRegressor(random_state=random_state),
        "mlp": MLPRegressor(hidden_layer_sizes=(64,32), max_iter=800, random_state=random_state)
    }
    if model_key not in models:
        raise ValueError(f"Unknown model_key: {model_key}")

    reg = Pipeline(steps=[("pre", pre), ("model", models[model_key])])
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)

    metrics = {
        "mae": float(mean_absolute_error(yte, pred)),
        "rmse": float(mean_squared_error(yte, pred, squared=False)),
        "r2": float(r2_score(yte, pred))
    }
    notes = "Pipeline: OneHotEncoder + StandardScaler + model. Metrics computed on test split."
    rep = TrainReport(task="regression", model_name=model_key, metrics=metrics, notes=notes)
    extra = {"y_true": yte.to_numpy(), "y_pred": pred}
    return reg, rep, extra

def compute_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def tune_classifier(df: pd.DataFrame, target: str, model_key: str, random_state: int = 42):
    X,y = split_xy(df, target)
    pre = build_preprocess(X)
    if model_key == "logreg":
        model = LogisticRegression(max_iter=3000)
        params = {"model__C": [0.1,1.0,10.0]}
    elif model_key == "rf":
        model = RandomForestClassifier(random_state=random_state)
        params = {"model__n_estimators": [100,200], "model__max_depth": [None,6,10]}
    else:
        model = DecisionTreeClassifier(random_state=random_state)
        params = {"model__max_depth": [3,5,7,None]}
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    gs = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
    gs.fit(X,y)
    return gs
