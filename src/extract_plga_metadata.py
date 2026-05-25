import argparse
import csv
from pathlib import Path
from typing import Any

from import_json_datasets import find_release_curves, first_number, load_json_payload, text_value


OUTPUT_COLUMNS = [
    "paper_id",
    "paper_title",
    "doi",
    "year",
    "formulation_id",
    "carrier_type",
    "carrier_class",
    "drug_name",
    "manufacturing_method",
    "release_method",
    "release_medium",
    "release_pH_values",
    "release_temperature_C",
    "particle_size_nm",
    "zeta_potential_mv",
    "pdi",
    "drug_to_carrier_mass_ratio",
    "polymer_to_drug_ratio",
    "drug_loading_content_percent",
    "encapsulation_efficiency_percent",
    "release_points_count",
    "ml_usable_points_count",
    "recommended_exclusion_from_release_training",
    "metadata_use",
    "notes",
    "source_file",
]


def first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = text_value(payload.get(key))
        if value:
            return value
    return ""


def base_metadata(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    fixed = payload.get("fixed_conditions", {}) if isinstance(payload.get("fixed_conditions"), dict) else {}
    carrier = payload.get("carrier_formulation", {}) if isinstance(payload.get("carrier_formulation"), dict) else {}
    loading = payload.get("loading_information", {}) if isinstance(payload.get("loading_information"), dict) else {}
    properties = (
        payload.get("reported_particle_properties", {})
        if isinstance(payload.get("reported_particle_properties"), dict)
        else {}
    )
    quality = payload.get("quality_control", {}) if isinstance(payload.get("quality_control"), dict) else {}
    drug = payload.get("drug", {}) if isinstance(payload.get("drug"), dict) else {}
    release_curves = find_release_curves(payload)
    release_points = sum(
        len(curve.get("data_points", []))
        for curve in release_curves
        if isinstance(curve.get("data_points"), list)
    )

    return {
        "paper_id": text_value(payload.get("paper_id")) or path.stem,
        "paper_title": first_text(payload, "paper_title", "research_title", "title") or path.stem,
        "doi": text_value(payload.get("doi")),
        "year": first_number(payload.get("year")),
        "formulation_id": "paper_level",
        "carrier_type": text_value(carrier.get("carrier_type")),
        "carrier_class": text_value(carrier.get("carrier_class")),
        "drug_name": text_value(payload.get("drug_name")) or text_value(drug.get("name")),
        "manufacturing_method": text_value(fixed.get("manufacturing_method"))
        or text_value(carrier.get("manufacturing_method")),
        "release_method": text_value(fixed.get("release_method")),
        "release_medium": text_value(fixed.get("release_medium")),
        "release_pH_values": text_value(fixed.get("release_pH_values")),
        "release_temperature_C": first_number(fixed.get("release_temperature_C")),
        "particle_size_nm": first_number(properties.get("particle_size_nm")),
        "zeta_potential_mv": first_number(properties.get("zeta_potential_mV") or properties.get("zeta_potential_mv")),
        "pdi": first_number(properties.get("pdi")),
        "drug_to_carrier_mass_ratio": first_number(loading.get("drug_to_carrier_mass_ratio")),
        "polymer_to_drug_ratio": first_number(loading.get("polymer_to_drug_ratio")),
        "drug_loading_content_percent": first_number(loading.get("drug_loading_content_percent")),
        "encapsulation_efficiency_percent": first_number(loading.get("encapsulation_efficiency_percent")),
        "release_points_count": release_points,
        "ml_usable_points_count": first_number(quality.get("ml_usable_points_count")),
        "recommended_exclusion_from_release_training": text_value(quality.get("recommended_exclusion")),
        "metadata_use": "reference_only" if release_points == 0 else "reference_plus_release_training_available",
        "notes": text_value(quality.get("exclusion_reason")) or text_value(payload.get("carrier_summary")),
        "source_file": str(path),
    }


def formulation_rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    base = base_metadata(payload, path)
    formulations = payload.get("reported_particle_properties_by_formulation")
    if not isinstance(formulations, list) or not formulations:
        return [base]

    rows: list[dict[str, Any]] = []
    for index, formulation in enumerate(formulations, start=1):
        if not isinstance(formulation, dict):
            continue
        row = dict(base)
        row.update(
            {
                "formulation_id": text_value(formulation.get("formulation_id")) or f"formulation_{index}",
                "carrier_type": text_value(formulation.get("carrier_type")) or base["carrier_type"],
                "particle_size_nm": first_number(
                    formulation.get("particle_size_nm") or formulation.get("nominal_particle_size_nm")
                )
                or base["particle_size_nm"],
                "zeta_potential_mv": first_number(
                    formulation.get("zeta_potential_mV") or formulation.get("zeta_potential_mv")
                )
                or base["zeta_potential_mv"],
                "pdi": first_number(formulation.get("pdi")) or base["pdi"],
                "drug_loading_content_percent": first_number(formulation.get("drug_loading_content_percent"))
                or base["drug_loading_content_percent"],
                "encapsulation_efficiency_percent": first_number(
                    formulation.get("encapsulation_efficiency_percent")
                )
                or base["encapsulation_efficiency_percent"],
                "notes": text_value(formulation.get("source_note")) or base["notes"],
            }
        )
        rows.append(row)
    return rows or [base]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PLGA formulation metadata for reference-only use.")
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/plga_formulation_metadata.csv"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.json_files:
        payload = load_json_payload(path)
        if isinstance(payload, dict):
            rows.extend(formulation_rows(payload, path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})

    reference_only = sum(1 for row in rows if row["metadata_use"] == "reference_only")
    print(f"Saved metadata: {args.out}")
    print(f"Rows written: {len(rows)}")
    print(f"Reference-only rows: {reference_only}")


if __name__ == "__main__":
    main()
