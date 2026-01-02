"""
Command-line interface for the Risk Engine.

Usage:
    risk-engine train          Train the churn prediction model
    risk-engine predict        Score customers and show top risks
    risk-engine info           Show model and dataset information
"""

import argparse
import sys
from pathlib import Path


def cmd_train(args):
    """Train the XGBoost churn model."""
    from risk_engine.models.train_model import train_and_save_model

    print("Training churn prediction model...")
    train_and_save_model()
    print("Done.")


def cmd_predict(args):
    """Score customers and display top risks."""
    from risk_engine.models.train_model import get_model
    from risk_engine.data.preprocess import load_raw, split_features_target
    from risk_engine.data.feature_engineering import apply_feature_engineering

    print("Loading model and data...")
    model = get_model()
    df, _ = load_raw()
    df = apply_feature_engineering(df)
    X, y, id_cols = split_features_target(df)

    print("Scoring customers...")
    proba = model.predict_proba(X)[:, 1]
    df["churn_probability"] = proba

    # Show top N highest risk
    top_n = args.top if hasattr(args, "top") else 10
    top_risks = df.nlargest(top_n, "churn_probability")[
        ["churn_probability", "Contract", "tenure", "MonthlyCharges"]
    ]

    print(f"\nTop {top_n} highest risk customers:")
    print(top_risks.to_string(index=False))

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Total customers: {len(df)}")
    print(f"Mean churn probability: {proba.mean():.2%}")
    print(f"High risk (>50%): {(proba > 0.5).sum()} customers")


def cmd_info(args):
    """Show model and dataset information."""
    from risk_engine.models.train_model import get_model, MODELS_DIR
    from risk_engine.data.preprocess import load_raw

    print("Risk Engine - Model Information")
    print("=" * 40)

    # Model info
    model_path = MODELS_DIR / "model.pkl"
    if model_path.exists():
        import os
        from datetime import datetime

        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        size_kb = os.path.getsize(model_path) / 1024
        print(f"Model file: {model_path}")
        print(f"Last modified: {mtime.strftime('%Y-%m-%d %H:%M')}")
        print(f"Size: {size_kb:.1f} KB")

        model = get_model()
        if hasattr(model, "named_steps"):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")
    else:
        print("Model not found. Run 'risk-engine train' first.")

    print()

    # Dataset info
    print("Dataset Information")
    print("-" * 40)
    try:
        df, path = load_raw()
        print(f"Source: Kaggle (blastchar/telco-customer-churn)")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        if "Churn" in df.columns:
            churn_rate = (df["Churn"] == "Yes").mean()
            print(f"Churn rate: {churn_rate:.1%}")
    except Exception as e:
        print(f"Could not load dataset: {e}")


def main():
    parser = argparse.ArgumentParser(
        prog="risk-engine",
        description="Kaizar Risk Engine - Churn Prediction CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train command
    train_parser = subparsers.add_parser("train", help="Train the churn model")

    # predict command
    predict_parser = subparsers.add_parser("predict", help="Score customers")
    predict_parser.add_argument(
        "--top", type=int, default=10, help="Number of top risks to show (default: 10)"
    )

    # info command
    info_parser = subparsers.add_parser("info", help="Show model information")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
