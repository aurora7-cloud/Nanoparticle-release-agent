import argparse
import csv
import json
import math
from pathlib import Path


TARGET = "drug_release_percent"
NUMERIC_FEATURES = ["particle_size_nm", "zeta_potential_mv", "ph", "time_h"]
CATEGORICAL_FEATURES = ["carrier_type"]
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def to_float(value: str) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def load_dataset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for raw in reader:
            row = {column: raw.get(column, "") for column in raw}
            skip = False
            for column in NUMERIC_FEATURES + [TARGET]:
                value = to_float(row.get(column, ""))
                if value is None:
                    skip = True
                    break
                row[column] = value
            if skip:
                continue
            row[TARGET] = min(max(row[TARGET], 0.0), 100.0)
            row["carrier_type"] = str(row["carrier_type"]).strip()
            rows.append(row)

    if len(rows) < 10:
        raise ValueError("At least 10 usable rows are recommended for a first training run.")
    return rows


def numeric_ranges(rows: list[dict]) -> dict:
    ranges = {}
    for column in NUMERIC_FEATURES:
        values = [row[column] for row in rows]
        values_sorted = sorted(values)
        ranges[column] = {
            "min": min(values),
            "max": max(values),
            "median": values_sorted[len(values_sorted) // 2],
        }
    return ranges


def normalized_distance(a: dict, b: dict, ranges: dict) -> float:
    total = 0.0
    for column in NUMERIC_FEATURES:
        spread = max(ranges[column]["max"] - ranges[column]["min"], 1.0)
        total += ((a[column] - b[column]) / spread) ** 2

    if str(a["carrier_type"]).lower() != str(b["carrier_type"]).lower():
        total += 0.35
    return math.sqrt(total)


def predict_knn(query: dict, train_rows: list[dict], ranges: dict, k: int = 5) -> float:
    neighbors = sorted(
        ((normalized_distance(query, row, ranges), row) for row in train_rows),
        key=lambda item: item[0],
    )[:k]

    weighted_sum = 0.0
    weight_total = 0.0
    for distance, row in neighbors:
        weight = 1.0 / (distance + 0.05)
        weighted_sum += weight * row[TARGET]
        weight_total += weight
    return min(max(weighted_sum / weight_total, 0.0), 100.0)


def leave_one_out_metrics(rows: list[dict], ranges: dict, k: int) -> dict:
    actual = []
    predicted = []
    for index, row in enumerate(rows):
        train_rows = rows[:index] + rows[index + 1 :]
        prediction = predict_knn(row, train_rows, ranges, k=k)
        actual.append(row[TARGET])
        predicted.append(prediction)

    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    rmse = math.sqrt(mse)
    mean_actual = sum(actual) / len(actual)
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    return {"rmse": rmse, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dependency-free baseline drug release model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV dataset.")
    parser.add_argument("--model-out", type=Path, default=Path("models/drug_release_model.json"))
    parser.add_argument("--k", type=int, default=5, help="Number of nearest rows used for prediction.")
    args = parser.parse_args()

    rows = load_dataset(args.data)
    ranges = numeric_ranges(rows)
    carrier_types = sorted({row["carrier_type"] for row in rows})
    metrics = leave_one_out_metrics(rows, ranges, k=args.k)

    package = {
        "model_type": "weighted_knn_baseline",
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_ranges": ranges,
        "carrier_types": carrier_types,
        "k": args.k,
        "training_rows": rows,
        "metrics": metrics,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {args.model_out}")
    print(f"Rows used: {len(rows)}")
    print(f"Leave-one-out RMSE: {metrics['rmse']:.3f}")
    print(f"Leave-one-out R2: {metrics['r2']:.3f}")
    print("Note: this is a baseline agent. Upgrade to Random Forest after collecting real paper data.")


if __name__ == "__main__":
    main()
