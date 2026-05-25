import argparse
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

from train_rf_shap_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, add_theory_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict release with RF, SHAP, and theory-based interpretation.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--carrier-type", required=True)
    parser.add_argument("--drug-name", required=True)
    parser.add_argument("--release-medium", default="PBS buffer")
    parser.add_argument("--particle-size-nm", type=float)
    parser.add_argument("--zeta-potential-mv", type=float)
    parser.add_argument("--pdi", type=float)
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--temperature-C", type=float)
    parser.add_argument("--time-h", type=float, required=True)
    parser.add_argument("--drug-loading-content-percent", type=float)
    parser.add_argument("--encapsulation-efficiency-percent", type=float)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def build_query(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "carrier_type": args.carrier_type,
                "drug_name": args.drug_name,
                "release_medium": args.release_medium,
                "particle_size_nm": args.particle_size_nm,
                "zeta_potential_mv": args.zeta_potential_mv,
                "pdi": args.pdi,
                "ph": args.ph,
                "temperature_C": args.temperature_C,
                "time_h": args.time_h,
                "drug_loading_content_percent": args.drug_loading_content_percent,
                "encapsulation_efficiency_percent": args.encapsulation_efficiency_percent,
            }
        ]
    )


def training_range_warnings(query: pd.DataFrame, training: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for feature in NUMERIC_FEATURES:
        if feature not in query.columns or feature not in training.columns:
            continue
        value = pd.to_numeric(query.iloc[0].get(feature), errors="coerce")
        if pd.isna(value):
            warnings.append(f"{feature}: missing value was median-imputed.")
            continue
        observed = pd.to_numeric(training[feature], errors="coerce").dropna()
        if not observed.empty and (value < observed.min() or value > observed.max()):
            warnings.append(f"{feature}: {value:g} is outside training range {observed.min():g}-{observed.max():g}.")
    for feature in CATEGORICAL_FEATURES:
        value = str(query.iloc[0].get(feature) or "unknown")
        seen = set(training[feature].fillna("unknown").astype(str))
        if value not in seen:
            warnings.append(f"{feature}: '{value}' was not seen in the training set.")
    return warnings


def nearest_rows(query: pd.DataFrame, training: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    numeric = ["ph", "time_h", "particle_size_nm", "zeta_potential_mv", "drug_loading_content_percent"]
    q = query.iloc[0]
    distances = []
    for idx, row in training.iterrows():
        total = 0.0
        for feature in numeric:
            values = pd.to_numeric(training[feature], errors="coerce")
            span = max(float(values.max() - values.min()), 1.0) if values.notna().any() else 1.0
            qv = pd.to_numeric(pd.Series([q.get(feature)]), errors="coerce").iloc[0]
            rv = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
            if pd.isna(qv) or pd.isna(rv):
                total += 0.05
            else:
                total += ((float(qv) - float(rv)) / span) ** 2
        for feature in ["carrier_family", "carrier_type", "drug_name"]:
            if feature not in q.index or feature not in row.index:
                continue
            total += 0.15 if str(q.get(feature)).lower() != str(row.get(feature)).lower() else 0.0
        distances.append(math.sqrt(total))
    out = training.copy()
    out["distance"] = distances
    return out.sort_values("distance").head(n)


def korsmeyer_peppas_for_curve(curve: pd.DataFrame) -> dict | None:
    data = curve.sort_values("time_h")
    finite = data[(data["time_h"] > 0) & (data[TARGET] > 0)].copy()
    if len(finite) < 3:
        return None
    max_release = max(float(data[TARGET].max()), 1.0)
    finite["fraction"] = (finite[TARGET] / max_release).clip(1e-6, 0.999)
    early = finite[finite["fraction"] <= 0.6]
    if len(early) < 3:
        early = finite.head(3)
    x = np.log(early["time_h"].to_numpy(dtype=float))
    y = np.log(early["fraction"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, deg=1)
    n_value = float(slope)
    if n_value < 0.43:
        label = "Fickian diffusion/Higuchi compatibility is relatively plausible."
    elif n_value < 0.85:
        label = "Anomalous release is likely; diffusion may mix with swelling or carrier relaxation."
    else:
        label = "Case-II or erosion/relaxation-dominant behavior may be involved."
    return {"n": n_value, "k": float(math.exp(intercept)), "interpretation": label}


def theory_explanation(query: pd.Series, top_features: list[tuple[str, float]]) -> list[str]:
    names = " ".join(name for name, _ in top_features).lower()
    lines = []
    if "time" in names:
        lines.append("Time dominates cumulative release: this follows the Noyes-Whitney rate accumulating into Higuchi/Korsmeyer-Peppas style time dependence.")
    if "surface_area" in names or "particle_size" in names:
        lines.append("Particle size affects effective exposed area; smaller particles generally raise the area term A and can accelerate early release.")
    if "ph" in names or "acidic" in names:
        if float(query["ph"]) < 7.0:
            lines.append("Acidic pH can increase carrier swelling, ionization, degradation, or membrane destabilization, changing the release driving force.")
        else:
            lines.append("Near-neutral pH tends to preserve pH-sensitive carriers, so release is interpreted as more stability-limited.")
    if "zeta" in names:
        lines.append("Absolute zeta potential is a stability proxy: strong electrostatic repulsion reduces aggregation and helps maintain effective surface area.")
    if "loading" in names:
        lines.append("Drug loading acts as an initial concentration-gradient proxy; higher loading can increase the outward diffusion driving force.")
    if "carrier_type" in names or "drug_name" in names:
        lines.append("Carrier and drug identity proxy the real diffusion coefficient, matrix interactions, swelling, erosion, and binding strength not fully captured by numeric descriptors.")
    return lines


def main() -> None:
    args = parse_args()
    package = joblib.load(args.model)
    pipeline = package["pipeline"]
    training = package["training_frame"]
    query = add_theory_features(build_query(args))

    prediction = float(np.clip(pipeline.predict(query)[0], 0, 100))
    transformed_query = pipeline.named_steps["preprocessor"].transform(query)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = np.asarray(explainer.shap_values(transformed_query))[0]
    feature_names = package["feature_names"]
    ranked = sorted(zip(feature_names, shap_values), key=lambda item: abs(item[1]), reverse=True)[: args.top_n]

    nearest = nearest_rows(query, training, n=5)
    nearest_curve_key = nearest.iloc[0]["curve_group"] if not nearest.empty else None
    kp = None
    if nearest_curve_key:
        kp = korsmeyer_peppas_for_curve(training[training["curve_group"] == nearest_curve_key])

    warnings = training_range_warnings(query, training)
    print("RF + SHAP Drug Release Agent")
    print("=" * 34)
    print(f"Predicted cumulative release: {prediction:.2f}% at {args.time_h:g} h")
    print(f"SHAP baseline expected value: {package['expected_value']:.2f}%")
    print()
    print("Top local SHAP drivers:")
    for name, value in ranked:
        direction = "increased" if value >= 0 else "decreased"
        print(f"- {name}: {value:+.3f} percentage points ({direction} prediction)")
    print()
    print("Chemical-engineering interpretation:")
    for line in theory_explanation(query.iloc[0], ranked):
        print(f"- {line}")
    if kp:
        print(f"- Nearest-curve Korsmeyer-Peppas estimate: n={kp['n']:.3f}, k={kp['k']:.3f}. {kp['interpretation']}")
    if warnings:
        print()
        print("Reliability cautions:")
        for warning in warnings:
            print(f"- {warning}")
    print()
    print("Nearest training evidence:")
    for _, row in nearest.iterrows():
        print(
            f"- distance={row['distance']:.3f}, curve={row['curve_id']}, "
            f"paper={row['paper_id']}, time={row['time_h']} h, pH={row['ph']}, release={row[TARGET]}%"
        )


if __name__ == "__main__":
    main()
