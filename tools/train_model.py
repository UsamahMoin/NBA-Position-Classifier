#!/usr/bin/env python3
"""Train the NBA position classifier and export a browser-consumable artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "nba2021.csv"
OUTPUT_PATH = ROOT / "model-artifact.json"
RANDOM_STATE = 42
POSITION_LABELS = ["C", "PF", "PG", "SF", "SG"]


def clean_dataset() -> tuple[pd.DataFrame, int]:
    data = pd.read_csv(DATA_PATH)
    original_rows = len(data)

    for percentage in ["3P%", "FG%", "FT%"]:
        attempts = percentage.replace("%", "A")
        data.loc[data[attempts] == 0, percentage] = 0

    total_players = set(data.loc[data["Tm"] == "TOT", "Player"])
    data = data.loc[
        ~data["Player"].isin(total_players) | (data["Tm"] == "TOT")
    ].copy()

    return data.reset_index(drop=True), original_rows


def serialize_report(report: dict) -> dict:
    serialized = {}
    for label, values in report.items():
        if isinstance(values, dict):
            serialized[label] = {
                key: round(float(value), 6)
                for key, value in values.items()
            }
        else:
            serialized[label] = round(float(values), 6)
    return serialized


def train() -> dict:
    data, original_rows = clean_dataset()
    feature_frame = data.drop(columns=["Player", "Pos", "Tm"])
    target = data["Pos"]
    feature_frame = feature_frame.loc[:, feature_frame.var() != 0]

    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train_raw = feature_frame.iloc[train_indices]
    x_test_raw = feature_frame.iloc[test_indices]
    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_train_imputed = imputer.fit_transform(x_train_raw)
    x_test_imputed = imputer.transform(x_test_raw)
    x_train_scaled = scaler.fit_transform(x_train_imputed)
    x_test_scaled = scaler.transform(x_test_imputed)

    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_balanced, y_train_balanced = smote.fit_resample(
        x_train_scaled,
        y_train,
    )

    selector = RFECV(
        estimator=RandomForestClassifier(
            n_estimators=180,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="balanced_accuracy",
        min_features_to_select=5,
        n_jobs=1,
    )
    x_train_selected = selector.fit_transform(x_train_balanced, y_train_balanced)
    x_test_selected = selector.transform(x_test_scaled)

    parameter_grid = {
        "C": np.logspace(-2, 2, 10),
        "kernel": ["linear"],
    }
    search = GridSearchCV(
        SVC(decision_function_shape="ovr", random_state=RANDOM_STATE),
        parameter_grid,
        scoring="balanced_accuracy",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )
    search.fit(x_train_selected, y_train_balanced)
    model: SVC = search.best_estimator_

    train_predictions = model.predict(x_train_selected)
    test_predictions = model.predict(x_test_selected)
    selected_features = feature_frame.columns[selector.support_].tolist()
    selected_indices = np.flatnonzero(selector.support_)

    feature_importance_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced_subsample",
    )
    feature_importance_model.fit(x_train_selected, y_train_balanced)
    feature_importance = sorted(
        zip(selected_features, feature_importance_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    test_report = classification_report(
        y_test,
        test_predictions,
        labels=POSITION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, test_predictions, labels=POSITION_LABELS)

    raw_class_counts = (
        target.value_counts().reindex(POSITION_LABELS, fill_value=0).astype(int)
    )
    train_class_counts = (
        y_train.value_counts().reindex(POSITION_LABELS, fill_value=0).astype(int)
    )
    balanced_class_counts = (
        pd.Series(y_train_balanced)
        .value_counts()
        .reindex(POSITION_LABELS, fill_value=0)
        .astype(int)
    )

    # Linear SVC uses one-vs-one internally. Export all pairwise hyperplanes so
    # the browser can reproduce sklearn's multiclass vote and tie-break logic.
    pair_labels = []
    for first_index in range(len(model.classes_)):
        for second_index in range(first_index + 1, len(model.classes_)):
            pair_labels.append(
                [str(model.classes_[first_index]), str(model.classes_[second_index])]
            )

    players = []
    selected_mean = feature_frame[selected_features].mean()
    selected_std = feature_frame[selected_features].std().replace(0, 1)
    for row_index, row in data.iterrows():
        values = {
            feature: round(float(row[feature]), 6)
            for feature in selected_features
        }
        players.append(
            {
                "player": row["Player"],
                "position": row["Pos"],
                "team": row["Tm"],
                "values": values,
                "isTest": bool(row_index in set(test_indices)),
                "predicted": (
                    str(test_predictions[list(test_indices).index(row_index)])
                    if row_index in set(test_indices)
                    else None
                ),
            }
        )

    feature_stats = {
        feature: {
            "min": round(float(feature_frame[feature].min()), 6),
            "max": round(float(feature_frame[feature].max()), 6),
            "mean": round(float(selected_mean[feature]), 6),
            "std": round(float(selected_std[feature]), 6),
        }
        for feature in selected_features
    }
    position_profiles = {
        position: {
            feature: round(
                float(
                    data.loc[data["Pos"] == position, feature].mean()
                ),
                6,
            )
            for feature in selected_features
        }
        for position in POSITION_LABELS
    }

    return {
        "title": "NBA Position Classifier",
        "season": 2021,
        "originalRowCount": original_rows,
        "playerCount": len(data),
        "removedDuplicateRows": original_rows - len(data),
        "labels": POSITION_LABELS,
        "classCounts": raw_class_counts.to_dict(),
        "trainClassCounts": train_class_counts.to_dict(),
        "balancedClassCounts": balanced_class_counts.to_dict(),
        "split": {
            "train": len(train_indices),
            "test": len(test_indices),
            "testSize": 0.25,
            "randomState": RANDOM_STATE,
        },
        "metrics": {
            "trainingAccuracy": round(
                float(accuracy_score(y_train_balanced, train_predictions)),
                6,
            ),
            "testAccuracy": round(float(accuracy_score(y_test, test_predictions)), 6),
            "balancedAccuracy": round(
                float(balanced_accuracy_score(y_test, test_predictions)),
                6,
            ),
            "crossValidationBalancedAccuracy": round(
                float(search.best_score_),
                6,
            ),
            "bestC": round(float(search.best_params_["C"]), 8),
        },
        "classificationReport": serialize_report(test_report),
        "confusionMatrix": confusion.astype(int).tolist(),
        "allFeatures": feature_frame.columns.tolist(),
        "selectedFeatures": selected_features,
        "selectedFeatureIndices": selected_indices.astype(int).tolist(),
        "featureImportance": [
            {"feature": feature, "importance": round(float(importance), 8)}
            for feature, importance in feature_importance
        ],
        "featureStats": feature_stats,
        "positionProfiles": position_profiles,
        "imputerStatistics": [
            round(float(value), 8)
            for value in imputer.statistics_[selected_indices]
        ],
        "scalerMean": [
            round(float(value), 8)
            for value in scaler.mean_[selected_indices]
        ],
        "scalerScale": [
            round(float(value), 8)
            for value in scaler.scale_[selected_indices]
        ],
        "model": {
            "classes": [str(label) for label in model.classes_],
            "pairLabels": pair_labels,
            "coefficients": [
                [round(float(value), 10) for value in row]
                for row in model.coef_
            ],
            "intercepts": [
                round(float(value), 10)
                for value in model.intercept_
            ],
        },
        "players": sorted(players, key=lambda player: player["player"]),
    }


if __name__ == "__main__":
    artifact = train()
    OUTPUT_PATH.write_text(
        json.dumps(artifact, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(json.dumps(artifact["metrics"], indent=2))
    print("Selected features:", ", ".join(artifact["selectedFeatures"]))
