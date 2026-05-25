import argparse
import json
import math
from pathlib import Path

from train_advanced_model import predict_ensemble, prepare_query, to_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict drug release with the curve-aware advanced agent.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--carrier-type", required=True)
    parser.add_argument("--drug-name", default="unknown")
    parser.add_argument("--release-medium", default="PBS")
    parser.add_argument("--particle-size-nm", type=float)
    parser.add_argument("--zeta-potential-mv", type=float)
    parser.add_argument("--pdi", type=float)
    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--time-h", type=float, required=True)
    parser.add_argument("--drug-loading-content-percent", type=float)
    parser.add_argument("--encapsulation-efficiency-percent", type=float)
    return parser.parse_args()


def build_query(args: argparse.Namespace) -> dict:
    return {
        "carrier_type": args.carrier_type,
        "drug_name": args.drug_name,
        "release_medium": args.release_medium,
        "particle_size_nm": args.particle_size_nm,
        "zeta_potential_mv": args.zeta_potential_mv,
        "pdi": args.pdi,
        "ph": args.ph,
        "time_h": args.time_h,
        "drug_loading_content_percent": args.drug_loading_content_percent,
        "encapsulation_efficiency_percent": args.encapsulation_efficiency_percent,
    }


def range_warnings(query: dict, package: dict) -> list[str]:
    stats = package["numeric_stats"]
    prepared = prepare_query(query, stats)
    warnings = []
    for column, limits in stats.items():
        if column in {"log_time_h", "sqrt_time_h"}:
            continue
        original = to_float(query.get(column))
        if original is None:
            warnings.append(f"{column} is missing and was imputed with median={limits['median']:.3g}.")
            continue
        if original < limits["min"] or original > limits["max"]:
            warnings.append(f"{column}={original:g} is outside training range ({limits['min']:g} to {limits['max']:g}).")
    if prepared["carrier_type"] not in package.get("carrier_types", []):
        warnings.append(f"carrier_type='{prepared['carrier_type']}' was not seen during training.")
    if prepared["drug_name"] not in package.get("drug_names", []):
        warnings.append(f"drug_name='{prepared['drug_name']}' was not seen during training.")
    return warnings


def reliability_label(warnings: list[str], nearest_curve_distance: float | None, nearest_row_distance: float | None) -> str:
    if len(warnings) >= 4:
        return "low"
    if warnings:
        return "medium-low"
    nearest = min(value for value in [nearest_curve_distance, nearest_row_distance] if value is not None)
    if nearest <= 0.20:
        return "high-for-screening"
    if nearest <= 0.45:
        return "medium-high"
    return "medium"


def mechanistic_notes(query: dict, prediction: float, result: dict, warnings: list[str]) -> str:
    lines = ["Interpretation:"]
    lines.append(f"- The advanced agent predicts about {prediction:.1f}% cumulative release at {query['time_h']:g} h.")
    curve_pred = result.get("curve_prediction")
    if curve_pred is not None:
        lines.append(
            f"- The final estimate blends point-neighbor prediction ({result['point_prediction']:.1f}%) "
            f"with curve-neighbor interpolation ({curve_pred:.1f}%)."
        )
    if query["ph"] <= 6.5:
        lines.append("- Acidic pH is often associated with faster release in pH-responsive systems.")
    elif query["ph"] >= 8.0:
        lines.append("- Basic pH can be important for PLGA/hydroxyl-FK866 behavior in the current dataset.")
    else:
        lines.append("- Near-neutral pH is interpreted against the closest full release curves.")
    if query.get("particle_size_nm") and query["particle_size_nm"] < 150:
        lines.append("- Smaller particle size may increase accessible surface area and support faster release.")
    if warnings:
        lines.append("- Caution: missing or out-of-range inputs reduce confidence; treat this as screening support.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    package = json.loads(args.model.read_text(encoding="utf-8"))
    query = build_query(args)
    result = predict_ensemble(
        query,
        package["training_rows"],
        package["curves"],
        package["numeric_stats"],
        int(package["k_rows"]),
        int(package["k_curves"]),
    )
    warnings = range_warnings(query, package)
    nearest_curve = result["curve_neighbors"][0][0] if result["curve_neighbors"] else None
    nearest_row = result["row_neighbors"][0][0] if result["row_neighbors"] else None
    reliability = reliability_label(warnings, nearest_curve, nearest_row)

    print("Advanced Drug Release Prediction Agent")
    print("=" * 42)
    print(f"Predicted drug release (%): {result['prediction']:.2f}")
    print(f"Reliability estimate: {reliability}")
    print(f"Blend weights: point_knn={result['blend']['point_knn']:.2f}, curve_knn={result['blend']['curve_knn']:.2f}")
    print()

    if warnings:
        print("Range and imputation checks:")
        for warning in warnings:
            print(f"- {warning}")
        print()

    print(mechanistic_notes(query, result["prediction"], result, warnings))

    if result["curve_neighbors"]:
        print()
        print("Nearest release curves:")
        for distance, curve, estimate in result["curve_neighbors"]:
            print(
                f"- distance={distance:.3f}, paper_id={curve.get('paper_id', '')}, "
                f"curve_id={curve.get('curve_id', '')}, carrier={curve.get('carrier_type', '')}, "
                f"drug={curve.get('drug_name', '')}, pH={curve.get('ph', '')}, "
                f"curve_estimate={estimate:.1f}%"
            )

    if result["row_neighbors"]:
        print()
        print("Nearest data points:")
        for distance, row in result["row_neighbors"][:5]:
            print(
                f"- distance={distance:.3f}, curve={row.get('curve_id', '')}, "
                f"time={row.get('time_h', '')} h, release={row.get('drug_release_percent', '')}%"
            )


if __name__ == "__main__":
    main()
