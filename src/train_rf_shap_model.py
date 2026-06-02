import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "drug_release_percent"
BASE_NUMERIC_FEATURES = [
    "particle_size_nm",
    "zeta_potential_mv",
    "ph",
    "time_h",
    "drug_loading_content_percent",
    "encapsulation_efficiency_percent",
]
THEORY_FEATURES = [
    "log_time_h",
    "sqrt_time_h",
    "surface_area_proxy_inv_nm",
    "acidic_trigger_7_4",
    "zeta_stability_abs_mv",
    "loading_gradient_proxy",
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + THEORY_FEATURES
CATEGORICAL_FEATURES = ["carrier_family", "carrier_type", "drug_name", "release_medium"]

THEORY_NOTES = {
    "noyes_whitney": "dM/dt = D*A*(Cs-C)/h: release rate is controlled by diffusion, exposed area, and concentration driving force.",
    "higuchi": "Mt = kH*sqrt(t): diffusion-controlled cumulative release often scales with square-root time.",
    "korsmeyer_peppas": "Mt/Minf = k*t^n: n helps flag whether simple Fickian diffusion is plausible or swelling/erosion/relaxation may contribute.",
    "surface_area_proxy_inv_nm": "For spherical particles at comparable mass, smaller diameter implies larger specific surface area, approximated here by 1/diameter.",
    "ph": "pH changes ionization/solubility and pH-sensitive carrier destabilization, altering the Noyes-Whitney driving force.",
    "zeta_potential": "Higher absolute zeta potential implies stronger electrostatic stabilization and less aggregation, preserving effective surface area.",
}


DRUG_ALIASES = {
    "5 dox": "Doxorubicin",
    "5-dox": "Doxorubicin",
    "dox": "Doxorubicin",
    "doxy": "Doxorubicin",
    "doxorubicin": "Doxorubicin",
    "doxorubicin hydrochloride": "Doxorubicin",
    "docetaxel": "Docetaxel",
    "dtx": "Docetaxel",
    "dtxl": "Docetaxel",
    "ptx": "Paclitaxel",
    "paclitaxel": "Paclitaxel",
    "5 fluorouracil": "5-Fluorouracil",
    "5-fluorouracil": "5-Fluorouracil",
    "5fu": "5-Fluorouracil",
    "5 fu": "5-Fluorouracil",
    "gemcitabine": "Gemcitabine",
    "insulin": "Insulin",
    "rifampicin": "Rifampicin",
}


def infer_carrier_family(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(column, "") or "")
        for column in ["carrier_type", "paper_title", "paper_id", "source_file"]
    ).lower()
    if "plga" in text:
        return "PLGA"
    if "chitosan" in text or "csnp" in text:
        return "chitosan"
    if "liposome" in text or "lipo" in text:
        return "liposome"
    return "other"


def normalize_category(column: str, value: object) -> str:
    if pd.isna(value):
        return "unknown buffer" if column == "release_medium" else "not_reported"
    text = str(value or "unknown").strip()
    lowered = text.lower().replace("_", " ").replace("-", " ")
    if not text or lowered in {"na", "n/a", "nan", "none", "null", "not reported", "not_reported", "unknown"}:
        return "unknown buffer" if column == "release_medium" else "not_reported"
    compact_alias = lowered.replace("(", " ").replace(")", " ")
    compact_alias = " ".join(compact_alias.split())
    if column == "drug_name":
        return DRUG_ALIASES.get(compact_alias, text)
    if column == "carrier_type" and ("liposome" in lowered or "lipo" in lowered):
        return "liposome"
    if column == "carrier_type" and "plga" in lowered:
        return "PLGA nanoparticle"
    if column == "carrier_type" and ("chitosan" in lowered or "csnp" in lowered):
        return "chitosan nanoparticle"
    if column == "release_medium":
        compact = lowered.replace(" ", "")
        if "not reported" in lowered or "not_reported" in str(value).lower() or "exact buffer species" in lowered:
            return "unknown buffer"
        if compact in {"pbs", "pbsbuffer", "phosphatebufferedsaline"} or "pbs" in compact:
            return "PBS"
        if "citric acid" in lowered and "na2hpo4" in lowered:
            return "citric acid/Na2HPO4 buffer"
        if "hcl" in lowered and "tris" in lowered:
            return "HCl/tris base buffer"
        if "acetate" in lowered:
            return "acetate buffer"
        if "phosphate" in lowered:
            return "phosphate buffer"
    return text


def normalize_carrier_with_family(row: pd.Series) -> str:
    carrier = str(row.get("carrier_type", "not_reported") or "not_reported")
    family = str(row.get("carrier_family", "other") or "other").lower()
    lowered = carrier.lower().strip()
    if lowered == "polymeric nanoparticle" and family == "plga":
        return "PLGA nanoparticle"
    if lowered == "nanoparticles" and family == "chitosan":
        return "chitosan nanoparticle"
    return carrier


def add_theory_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_h"] = pd.to_numeric(out["time_h"], errors="coerce").clip(lower=0)
    out["particle_size_nm"] = pd.to_numeric(out["particle_size_nm"], errors="coerce")
    out["zeta_potential_mv"] = pd.to_numeric(out["zeta_potential_mv"], errors="coerce")
    out["ph"] = pd.to_numeric(out["ph"], errors="coerce")
    out["drug_loading_content_percent"] = pd.to_numeric(out["drug_loading_content_percent"], errors="coerce")

    out["log_time_h"] = np.log1p(out["time_h"])
    out["sqrt_time_h"] = np.sqrt(out["time_h"])
    out["surface_area_proxy_inv_nm"] = 1.0 / out["particle_size_nm"].where(out["particle_size_nm"] > 0)
    out["acidic_trigger_7_4"] = (7.4 - out["ph"]).clip(lower=0)
    out["zeta_stability_abs_mv"] = out["zeta_potential_mv"].abs()
    out["loading_gradient_proxy"] = out["drug_loading_content_percent"]
    if "carrier_family" not in out.columns:
        out["carrier_family"] = out.apply(infer_carrier_family, axis=1)
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].map(lambda value: normalize_category(column, value))
    out["carrier_type"] = out.apply(normalize_carrier_with_family, axis=1)
    return out


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET, "time_h", "ph"]).copy()
    df[TARGET] = df[TARGET].clip(0, 100)
    df["curve_group"] = df["paper_id"].astype(str) + "::" + df["curve_id"].astype(str)
    return add_theory_features(df)


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    forest = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", forest)])


def cross_validate(df: pd.DataFrame) -> dict:
    groups = df["curve_group"].to_numpy()
    unique_groups = np.unique(groups)
    splitter = LeaveOneGroupOut() if len(unique_groups) <= 8 else GroupKFold(n_splits=min(5, len(unique_groups)))
    y_true: list[float] = []
    y_pred: list[float] = []
    for train_idx, test_idx in splitter.split(df, df[TARGET], groups):
        fold_model = build_pipeline()
        fold_model.fit(df.iloc[train_idx], df.iloc[train_idx][TARGET])
        pred = fold_model.predict(df.iloc[test_idx])
        y_true.extend(df.iloc[test_idx][TARGET].to_list())
        y_pred.extend(np.clip(pred, 0, 100).tolist())
    return {
        "rows": int(len(df)),
        "curves": int(len(unique_groups)),
        "papers": int(df["paper_id"].nunique()),
        "group_cv_rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "group_cv_mae": float(mean_absolute_error(y_true, y_pred)),
        "group_cv_r2": float(r2_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
    }


def aggregate_shap(feature_names: list[str], values: np.ndarray) -> list[dict]:
    mean_abs = np.abs(values).mean(axis=0)
    rows = sorted(zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "mean_abs_shap": float(score)} for name, score in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest drug-release model with SHAP analysis.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models/rf_shap_drug_release_model.joblib"))
    parser.add_argument("--summary-out", type=Path, default=Path("outputs/rf_shap_summary.json"))
    args = parser.parse_args()

    df = load_dataset(args.data)
    pipeline = build_pipeline()
    metrics = cross_validate(df)
    pipeline.fit(df, df[TARGET])

    transformed = pipeline.named_steps["preprocessor"].transform(df)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(transformed)

    package = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "training_columns": df.columns.tolist(),
        "training_frame": df,
        "theory_notes": THEORY_NOTES,
        "expected_value": float(np.ravel(explainer.expected_value)[0]),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, args.model_out)

    summary = {
        "model_type": "RandomForestRegressor + SHAP TreeExplainer",
        "data_file": str(args.data),
        "model_file": str(args.model_out),
        "metrics": metrics,
        "oob_score": float(pipeline.named_steps["model"].oob_score_),
        "global_shap_top20": aggregate_shap(feature_names, np.asarray(shap_values))[:20],
        "theory_notes": THEORY_NOTES,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
