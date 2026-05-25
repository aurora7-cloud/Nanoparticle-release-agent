import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "paper_id",
    "paper_title",
    "curve_id",
    "figure",
    "carrier_type",
    "drug_name",
    "particle_size_nm",
    "zeta_potential_mv",
    "pdi",
    "ph",
    "temperature_C",
    "release_medium",
    "concentration_value",
    "concentration_unit",
    "time_h",
    "drug_release_percent",
    "drug_loading_content_percent",
    "encapsulation_efficiency_percent",
    "extraction_method",
    "reliability_score",
    "notes",
    "source_file",
]


CONFIDENCE_SCORE = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


JSON_LITERAL_FIXES = {
    "True": "true",
    "False": "false",
    "None": "null",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug[:80] or "paper"


def first_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "value" in value:
            return first_number(value.get("value"))
        for item in value.values():
            number = first_number(item)
            if number is not None:
                return number
        return None
    text = str(value).strip()
    if not text or text.lower() in {"not_reported", "not reported", "na", "n/a", "none", "null"}:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if "value" in value:
            return str(value["value"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "; ".join(text_value(item) for item in value if text_value(item))
    text = str(value).strip()
    return "" if text.lower() in {"not_reported", "not reported", "none", "null"} else text


def explicitly_not_applicable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        if "value" in value:
            return explicitly_not_applicable(value.get("value"))
        return False
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text in {"not_applicable", "n_a", "na"}


def curve_number_or_base(curve: dict[str, Any], base: dict[str, Any], keys: tuple[str, ...], base_key: str) -> float | None:
    for key in keys:
        if key not in curve:
            continue
        value = curve.get(key)
        if explicitly_not_applicable(value):
            return None
        number = first_number(value)
        if number is not None:
            return number
    return base[base_key]


def parse_point(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        return point
    text = str(point).strip()
    if text.startswith("@{") and text.endswith("}"):
        text = text[2:-1]
    parsed: dict[str, Any] = {}
    for part in text.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def find_release_curves(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("release_curves"), list):
            return [curve for curve in payload["release_curves"] if isinstance(curve, dict)]
        curves: list[dict[str, Any]] = []
        for value in payload.values():
            curves.extend(find_release_curves(value))
        return curves
    if isinstance(payload, list):
        curves = []
        for item in payload:
            curves.extend(find_release_curves(item))
        return curves
    return []


def load_json_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(
            r"\b(True|False|None)\b",
            lambda match: JSON_LITERAL_FIXES[match.group(1)],
            text,
        )
        return json.loads(fixed)


def metadata(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    title = (
        payload.get("paper_title")
        or payload.get("research_title")
        or payload.get("title")
        or payload.get("drug_delivery_system_analysis", {}).get("research_title")
        or source_path.stem
    )
    fixed = payload.get("fixed_conditions", {})
    release_conditions = fixed.get("release_conditions", {}) if isinstance(fixed, dict) else {}
    loading = payload.get("loading_information", {}) if isinstance(payload.get("loading_information"), dict) else {}
    reported = payload.get("reported_particle_properties", {}) if isinstance(payload.get("reported_particle_properties"), dict) else {}
    return {
        "paper_title": str(title),
        "paper_id": text_value(payload.get("paper_id")) or slugify(str(title)),
        "drug_name": payload.get("drug") or payload.get("drug_name") or "",
        "release_medium": (release_conditions.get("buffer") or fixed.get("release_medium") or "") if isinstance(fixed, dict) else "",
        "concentration_value": first_number(loading.get("release_concentration")),
        "concentration_unit": "",
        "particle_size_nm": first_number(reported.get("particle_size_nm")),
        "zeta_potential_mv": first_number(reported.get("zeta_potential_mV") or reported.get("zeta_potential_mv")),
        "pdi": first_number(reported.get("pdi")),
        "drug_loading_content_percent": first_number(loading.get("drug_loading_content_percent")),
        "encapsulation_efficiency_percent": first_number(loading.get("encapsulation_efficiency_percent")),
    }


def flatten_file(path: Path) -> list[dict[str, Any]]:
    payload = load_json_payload(path)
    base = metadata(payload if isinstance(payload, dict) else {}, path)
    rows: list[dict[str, Any]] = []

    for index, curve in enumerate(find_release_curves(payload), start=1):
        curve_id = text_value(curve.get("curve_id")) or f"curve_{index}"
        notes = "; ".join(
            item for item in [
                text_value(curve.get("figure")),
                text_value(curve.get("notes")),
            ] if item
        )
        curve_defaults = {
            "paper_id": base["paper_id"],
            "paper_title": base["paper_title"],
            "curve_id": curve_id,
            "figure": text_value(curve.get("figure")),
            "carrier_type": text_value(curve.get("carrier_type")),
            "drug_name": text_value(curve.get("drug_name")) or text_value(base["drug_name"]),
            "particle_size_nm": curve_number_or_base(curve, base, ("particle_size_nm",), "particle_size_nm"),
            "zeta_potential_mv": curve_number_or_base(curve, base, ("zeta_potential_mV", "zeta_potential_mv"), "zeta_potential_mv"),
            "pdi": curve_number_or_base(curve, base, ("pdi",), "pdi"),
            "ph": first_number(curve.get("pH") or curve.get("ph")),
            "temperature_C": first_number(curve.get("temperature_C") or curve.get("temperature_c")),
            "release_medium": text_value(curve.get("release_medium")) or text_value(base["release_medium"]),
            "concentration_value": first_number(curve.get("concentration_value")) or base["concentration_value"],
            "concentration_unit": text_value(curve.get("concentration_unit")) or base["concentration_unit"],
            "drug_loading_content_percent": curve_number_or_base(curve, base, ("drug_loading_content_percent",), "drug_loading_content_percent"),
            "encapsulation_efficiency_percent": curve_number_or_base(curve, base, ("encapsulation_efficiency_percent",), "encapsulation_efficiency_percent"),
            "notes": notes,
            "source_file": str(path),
        }

        for raw_point in curve.get("data_points", []):
            point = parse_point(raw_point)
            time_h = first_number(point.get("time_h"))
            release = first_number(point.get("drug_release_percent"))
            if time_h is None or release is None:
                continue
            confidence = text_value(point.get("confidence")).lower()
            row = dict(curve_defaults)
            row.update(
                {
                    "time_h": time_h,
                    "drug_release_percent": min(max(release, 0.0), 100.0),
                    "extraction_method": text_value(point.get("extraction_type")) or "json_extracted",
                    "reliability_score": CONFIDENCE_SCORE.get(confidence, ""),
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten paper-level release-curve JSON files into one CSV.")
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/combined_drug_release_dataset.csv"))
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.json_files:
        rows.extend(flatten_file(path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})

    print(f"Saved dataset: {args.out}")
    print(f"Rows written: {len(rows)}")
    print(f"Curves written: {len({(row['paper_id'], row['curve_id']) for row in rows})}")


if __name__ == "__main__":
    main()
