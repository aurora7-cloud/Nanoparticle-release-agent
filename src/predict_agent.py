import argparse
import csv
import json
import math
from pathlib import Path


NUMERIC_FEATURES = ["particle_size_nm", "zeta_potential_mv", "ph", "time_h"]
TARGET = "drug_release_percent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict drug release and explain the result.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained JSON model.")
    parser.add_argument("--data", type=Path, help="Optional CSV path for extra range checks.")
    parser.add_argument("--carrier-type", required=True)
    parser.add_argument("--particle-size-nm", type=float, required=True)
    parser.add_argument("--zeta-potential-mv", type=float, required=True)
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--time-h", type=float, required=True)
    return parser.parse_args()


def build_query(args: argparse.Namespace) -> dict:
    return {
        "carrier_type": args.carrier_type,
        "particle_size_nm": args.particle_size_nm,
        "zeta_potential_mv": args.zeta_potential_mv,
        "ph": args.ph,
        "time_h": args.time_h,
    }


def normalized_distance(a: dict, b: dict, ranges: dict) -> float:
    total = 0.0
    for column in NUMERIC_FEATURES:
        spread = max(ranges[column]["max"] - ranges[column]["min"], 1.0)
        total += ((float(a[column]) - float(b[column])) / spread) ** 2
    if str(a["carrier_type"]).lower() != str(b["carrier_type"]).lower():
        total += 0.35
    return math.sqrt(total)


def predict_knn(query: dict, package: dict) -> tuple[float, list[tuple[float, dict]]]:
    neighbors = sorted(
        (
            (normalized_distance(query, row, package["numeric_ranges"]), row)
            for row in package["training_rows"]
        ),
        key=lambda item: item[0],
    )[: int(package["k"])]

    weighted_sum = 0.0
    weight_total = 0.0
    for distance, row in neighbors:
        weight = 1.0 / (distance + 0.05)
        weighted_sum += weight * float(row[TARGET])
        weight_total += weight
    prediction = min(max(weighted_sum / weight_total, 0.0), 100.0)
    return prediction, neighbors


def range_warnings(query: dict, package: dict) -> list[str]:
    warnings = []
    for column, limits in package["numeric_ranges"].items():
        value = float(query[column])
        if value < limits["min"] or value > limits["max"]:
            warnings.append(
                f"{column}={value:g} is outside training range "
                f"({limits['min']:g} to {limits['max']:g})."
            )

    if query["carrier_type"] not in package["carrier_types"]:
        warnings.append(f"carrier_type='{query['carrier_type']}' was not seen during training.")
    return warnings


def reliability_label(warnings: list[str], nearest_distance: float) -> str:
    if warnings:
        return "low" if len(warnings) >= 3 else "medium-low"
    if nearest_distance <= 0.25:
        return "medium-high"
    return "medium"


def load_csv_rows(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def explain(query: dict, prediction: float, warnings: list[str], neighbors: list[tuple[float, dict]]) -> str:
    lines = []
    lines.append("Interpretation:")
    lines.append(
        f"- The agent predicts approximately {prediction:.1f}% cumulative drug release "
        f"at {query['time_h']:g} h."
    )

    if query["ph"] <= 6.5:
        lines.append("- Acidic pH may support faster release for pH-responsive carrier systems.")
    else:
        lines.append("- Neutral/basic pH may indicate a more moderate release environment.")

    if query["particle_size_nm"] <= 130:
        lines.append("- Smaller particle size can increase surface area, which may support faster release.")
    elif query["particle_size_nm"] >= 180:
        lines.append("- Larger particle size may slow release compared with smaller particles.")
    else:
        lines.append("- Particle size is in a middle range, so carrier chemistry and pH should be interpreted together.")

    if abs(query["zeta_potential_mv"]) >= 25:
        lines.append("- High absolute zeta potential suggests stronger colloidal stability, which can affect release behavior.")
    else:
        lines.append("- Moderate zeta potential should be interpreted together with carrier type and medium.")

    if warnings:
        lines.append("- Caution: part of this input is outside the training data, so use it only for rough screening.")
    else:
        lines.append("- The input is inside the training ranges, so the result is more suitable as a screening estimate.")

    if neighbors:
        releases = [float(row[TARGET]) for _, row in neighbors]
        lines.append(
            f"- Similar training rows had release values around {min(releases):.1f}% to {max(releases):.1f}%."
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    package = json.loads(args.model.read_text(encoding="utf-8"))
    query = build_query(args)
    prediction, neighbors = predict_knn(query, package)
    warnings = range_warnings(query, package)
    label = reliability_label(warnings, neighbors[0][0] if neighbors else 999.0)

    print("Drug Release Prediction Agent")
    print("=" * 32)
    print(f"Predicted drug release (%): {prediction:.2f}")
    print(f"Reliability estimate: {label}")
    print()

    if warnings:
        print("Range checks:")
        for warning in warnings:
            print(f"- {warning}")
        print()

    print(explain(query, prediction, warnings, neighbors))

    if neighbors:
        print()
        print("Nearest rows used for sanity check:")
        for distance, row in neighbors:
            print(
                f"- distance={distance:.3f}, paper_id={row.get('paper_id', '')}, "
                f"carrier={row.get('carrier_type', '')}, size={row.get('particle_size_nm', '')}, "
                f"zeta={row.get('zeta_potential_mv', '')}, pH={row.get('ph', '')}, "
                f"time={row.get('time_h', '')}, release={row.get(TARGET, '')}%"
            )


if __name__ == "__main__":
    main()
