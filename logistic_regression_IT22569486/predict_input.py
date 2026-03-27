from pathlib import Path
import argparse
import json

import joblib
import numpy as np


# python .\predict_input.py --values '406,-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919,0'
# python predict_input.py --values '0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62'    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict credit card fraud for one input row using saved model artifacts."
    )
    parser.add_argument(
        "--values",
        type=str,
        help=(
            "Comma-separated record in model order. Accepts either features-only "
            "or features+label as the last value (e.g., ...,149.62,0)."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_input",
        type=str,
        help=(
            "JSON object with feature names and numeric values, "
            "for example: '{\"Time\":0,\"V1\":-1.2,...,\"Amount\":100}'"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: precision_threshold from metadata).",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Print feature order expected by the model and exit.",
    )
    return parser.parse_args()


def default_feature_columns(count: int) -> list[str]:
    # Common schema for the credit card fraud dataset: Time, V1..V28, Amount.
    if count == 30:
        return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return [f"feature_{idx + 1}" for idx in range(count)]


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def collect_values_interactively(feature_columns: list[str]) -> list[float]:
    values: list[float] = []
    print("Enter one value for each feature:")
    for feature in feature_columns:
        while True:
            raw_value = input(f"{feature}: ").strip()
            try:
                values.append(float(raw_value))
                break
            except ValueError:
                print("Please enter a valid numeric value.")
    return values


def parse_values_input(values_input: str, expected_count: int) -> list[float]:
    def _clean_token(token: str) -> str:
        cleaned = token.strip()
        if (
            len(cleaned) >= 2
            and cleaned[0] == cleaned[-1]
            and cleaned[0] in {"\"", "'"}
        ):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    raw_values = [_clean_token(part) for part in values_input.split(",")]
    raw_values = [part for part in raw_values if part != ""]

    if len(raw_values) == expected_count + 1:
        # If a full dataset row is pasted, ignore trailing class label.
        raw_values = raw_values[:-1]
    elif len(raw_values) != expected_count:
        raise ValueError(
            f"Expected {expected_count} feature values (or {expected_count + 1} "
            f"values including a trailing label), but got {len(raw_values)} in --values."
        )

    return [float(value) for value in raw_values]


def parse_json_input(json_input: str, feature_columns: list[str]) -> list[float]:
    payload = json.loads(json_input)
    if not isinstance(payload, dict):
        raise ValueError("--json must be a JSON object mapping feature names to values.")

    missing = [feature for feature in feature_columns if feature not in payload]
    if missing:
        raise ValueError(
            "Missing feature(s) in JSON input: " + ", ".join(missing)
        )

    return [float(payload[feature]) for feature in feature_columns]


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "logistic_model.joblib"
    scaler_path = base_dir / "logistic_scaler.joblib"
    metadata_path = base_dir / "logistic_model_meta.json"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run logistic_regression.py first to create "
            "logistic_model.joblib and logistic_scaler.joblib."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = load_metadata(metadata_path)

    feature_count = int(getattr(model, "n_features_in_", 0))
    if feature_count <= 0:
        raise ValueError("Could not determine expected number of input features from model.")

    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or len(feature_columns) != feature_count:
        feature_columns = default_feature_columns(feature_count)

    if args.show_features:
        print("Expected feature order:")
        for index, feature in enumerate(feature_columns, start=1):
            print(f"{index:>2}. {feature}")
        return

    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        threshold = float(metadata.get("precision_threshold", 0.5))

    if args.values and args.json_input:
        raise ValueError("Use either --values or --json, not both.")

    if args.values:
        values = parse_values_input(args.values, feature_count)
    elif args.json_input:
        values = parse_json_input(args.json_input, feature_columns)
    else:
        values = collect_values_interactively(feature_columns)

    sample = np.array(values, dtype=float).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    fraud_probability = float(model.predict_proba(sample_scaled)[0, 1])
    prediction = int(fraud_probability >= threshold)

    print("=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Threshold: {threshold:.4f}")
    print(f"Fraud probability (Class 1): {fraud_probability:.6f}")
    print(f"Predicted class: {prediction} ({'Fraud' if prediction == 1 else 'Not Fraud'})")


if __name__ == "__main__":
    main()
