import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


TARGET = "drug_release_percent"
NUMERIC_FEATURES = [
    "particle_size_nm",
    "zeta_potential_mv",
    "pdi",
    "ph",
    "time_h",
    "log_time_h",
    "sqrt_time_h",
    "drug_loading_content_percent",
    "encapsulation_efficiency_percent",
]
CATEGORICAL_FEATURES = ["carrier_type", "drug_name", "release_medium"]
CURVE_FEATURES = [
    "particle_size_nm",
    "zeta_potential_mv",
    "pdi",
    "ph",
    "drug_loading_content_percent",
    "encapsulation_efficiency_percent",
]


def to_float(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for raw in reader:
            release = to_float(raw.get(TARGET))
            time_h = to_float(raw.get("time_h"))
            ph = to_float(raw.get("ph"))
            if release is None or time_h is None or ph is None:
                continue
            row = {key: (value.strip() if isinstance(value, str) else value) for key, value in raw.items()}
            row[TARGET] = min(max(release, 0.0), 100.0)
            row["time_h"] = max(time_h, 0.0)
            row["log_time_h"] = math.log1p(row["time_h"])
            row["sqrt_time_h"] = math.sqrt(row["time_h"])
            for column in NUMERIC_FEATURES:
                if column not in {"time_h", "log_time_h", "sqrt_time_h"}:
                    row[column] = to_float(row.get(column))
            for column in CATEGORICAL_FEATURES:
                row[column] = str(row.get(column) or "unknown").strip() or "unknown"
            row["curve_key"] = f"{row.get('paper_id', '')}::{row.get('curve_id', '')}"
            rows.append(row)
    if len(rows) < 10:
        raise ValueError("At least 10 usable rows are needed for the advanced model.")
    return rows


def build_stats(rows: list[dict]) -> dict:
    stats = {}
    for column in NUMERIC_FEATURES:
        values = [row[column] for row in rows if row.get(column) is not None]
        if not values:
            stats[column] = {"min": 0.0, "max": 1.0, "median": 0.0}
            continue
        values_sorted = sorted(values)
        stats[column] = {
            "min": min(values),
            "max": max(values),
            "median": median(values_sorted),
        }
    return stats


def prepare_query(row: dict, stats: dict) -> dict:
    prepared = dict(row)
    prepared["time_h"] = float(prepared.get("time_h") or 0.0)
    prepared["log_time_h"] = math.log1p(prepared["time_h"])
    prepared["sqrt_time_h"] = math.sqrt(prepared["time_h"])
    for column in NUMERIC_FEATURES:
        value = to_float(prepared.get(column))
        prepared[column] = stats[column]["median"] if value is None else value
    for column in CATEGORICAL_FEATURES:
        prepared[column] = str(prepared.get(column) or "unknown").strip() or "unknown"
    return prepared


def numeric_distance(a: dict, b: dict, stats: dict, columns: list[str]) -> float:
    total = 0.0
    for column in columns:
        spread = max(stats[column]["max"] - stats[column]["min"], 1.0)
        av = to_float(a.get(column))
        bv = to_float(b.get(column))
        if av is None:
            av = stats[column]["median"]
            total += 0.05
        if bv is None:
            bv = stats[column]["median"]
            total += 0.05
        total += ((av - bv) / spread) ** 2
    return total


def row_distance(a: dict, b: dict, stats: dict) -> float:
    total = numeric_distance(a, b, stats, NUMERIC_FEATURES)
    for column in CATEGORICAL_FEATURES:
        if str(a.get(column, "")).lower() != str(b.get(column, "")).lower():
            total += 0.20
    return math.sqrt(total)


def curve_signature(rows: list[dict]) -> dict:
    first = rows[0]
    signature = {column: first.get(column) for column in CURVE_FEATURES + CATEGORICAL_FEATURES}
    signature["curve_key"] = first["curve_key"]
    signature["paper_id"] = first.get("paper_id", "")
    signature["curve_id"] = first.get("curve_id", "")
    signature["points"] = sorted(
        [{"time_h": row["time_h"], TARGET: row[TARGET]} for row in rows],
        key=lambda point: point["time_h"],
    )
    return signature


def build_curves(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["curve_key"]].append(row)
    return [curve_signature(curve_rows) for curve_rows in grouped.values() if len(curve_rows) >= 2]


def interpolate_curve(points: list[dict], time_h: float) -> float:
    ordered = sorted(points, key=lambda point: point["time_h"])
    if time_h <= ordered[0]["time_h"]:
        return ordered[0][TARGET]
    if time_h >= ordered[-1]["time_h"]:
        return ordered[-1][TARGET]
    else:
        a, b = ordered[0], ordered[-1]
        for left, right in zip(ordered, ordered[1:]):
            if left["time_h"] <= time_h <= right["time_h"]:
                a, b = left, right
                break
    span = max(b["time_h"] - a["time_h"], 1e-9)
    ratio = (time_h - a["time_h"]) / span
    return min(max(a[TARGET] + ratio * (b[TARGET] - a[TARGET]), 0.0), 100.0)


def curve_distance(query: dict, curve: dict, stats: dict) -> float:
    total = numeric_distance(query, curve, stats, CURVE_FEATURES)
    for column in CATEGORICAL_FEATURES:
        if str(query.get(column, "")).lower() != str(curve.get(column, "")).lower():
            total += 0.18
    time_h = float(query.get("time_h") or 0.0)
    curve_times = [point["time_h"] for point in curve["points"]]
    min_time = min(curve_times)
    max_time = max(curve_times)
    if time_h < min_time:
        total += min(math.log1p(min_time - time_h), 4.0) * 0.15
    elif time_h > max_time:
        total += min(math.log1p(time_h - max_time), 4.0) * 0.15
    return math.sqrt(total)


def predict_point_knn(query: dict, rows: list[dict], stats: dict, k: int) -> tuple[float, list[tuple[float, dict]]]:
    prepared = prepare_query(query, stats)
    neighbors = sorted(((row_distance(prepared, row, stats), row) for row in rows), key=lambda item: item[0])[:k]
    weighted_sum = 0.0
    weight_total = 0.0
    for distance, row in neighbors:
        weight = 1.0 / (distance + 0.05)
        weighted_sum += weight * row[TARGET]
        weight_total += weight
    return min(max(weighted_sum / weight_total, 0.0), 100.0), neighbors


def predict_curve_knn(query: dict, curves: list[dict], stats: dict, k: int) -> tuple[float | None, list[tuple[float, dict, float]]]:
    prepared = prepare_query(query, stats)
    neighbors = sorted(((curve_distance(prepared, curve, stats), curve) for curve in curves), key=lambda item: item[0])[:k]
    if not neighbors:
        return None, []
    weighted_sum = 0.0
    weight_total = 0.0
    enriched = []
    for distance, curve in neighbors:
        estimate = interpolate_curve(curve["points"], prepared["time_h"])
        weight = 1.0 / (distance + 0.05)
        weighted_sum += weight * estimate
        weight_total += weight
        enriched.append((distance, curve, estimate))
    return min(max(weighted_sum / weight_total, 0.0), 100.0), enriched


def predict_ensemble(query: dict, rows: list[dict], curves: list[dict], stats: dict, k_rows: int, k_curves: int) -> dict:
    point_prediction, row_neighbors = predict_point_knn(query, rows, stats, k_rows)
    curve_prediction, curve_neighbors = predict_curve_knn(query, curves, stats, k_curves)
    if curve_prediction is None:
        prediction = point_prediction
        blend = {"point_knn": 1.0, "curve_knn": 0.0}
    else:
        prediction = 0.45 * point_prediction + 0.55 * curve_prediction
        blend = {"point_knn": 0.45, "curve_knn": 0.55}
    return {
        "prediction": min(max(prediction, 0.0), 100.0),
        "point_prediction": point_prediction,
        "curve_prediction": curve_prediction,
        "blend": blend,
        "row_neighbors": row_neighbors,
        "curve_neighbors": curve_neighbors,
    }


def metrics(rows: list[dict], stats: dict, k_rows: int, k_curves: int) -> dict:
    curve_keys = sorted({row["curve_key"] for row in rows})
    predictions = []
    actuals = []
    for curve_key in curve_keys:
        train_rows = [row for row in rows if row["curve_key"] != curve_key]
        if len(train_rows) < k_rows:
            continue
        train_curves = build_curves(train_rows)
        for row in [row for row in rows if row["curve_key"] == curve_key]:
            result = predict_ensemble(row, train_rows, train_curves, stats, k_rows, k_curves)
            predictions.append(result["prediction"])
            actuals.append(row[TARGET])
    if not predictions:
        return {"grouped_rmse": None, "grouped_r2": None, "evaluated_rows": 0}
    mse = sum((a - p) ** 2 for a, p in zip(actuals, predictions)) / len(actuals)
    mean_actual = sum(actuals) / len(actuals)
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
    return {
        "grouped_rmse": math.sqrt(mse),
        "grouped_r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
        "evaluated_rows": len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a curve-aware drug release prediction model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("models/advanced_drug_release_model.json"))
    parser.add_argument("--k-rows", type=int, default=7)
    parser.add_argument("--k-curves", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.data)
    stats = build_stats(rows)
    curves = build_curves(rows)
    package = {
        "model_type": "curve_aware_knn_ensemble",
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "curve_features": CURVE_FEATURES,
        "numeric_stats": stats,
        "k_rows": args.k_rows,
        "k_curves": args.k_curves,
        "training_rows": rows,
        "curves": curves,
        "carrier_types": sorted({row["carrier_type"] for row in rows}),
        "drug_names": sorted({row["drug_name"] for row in rows}),
        "metrics": metrics(rows, stats, args.k_rows, args.k_curves),
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {args.model_out}")
    print(f"Rows used: {len(rows)}")
    print(f"Curves used: {len(curves)}")
    print(f"Grouped RMSE: {package['metrics']['grouped_rmse']:.3f}")
    print(f"Grouped R2: {package['metrics']['grouped_r2']:.3f}")


if __name__ == "__main__":
    main()
