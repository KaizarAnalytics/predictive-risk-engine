from pathlib import Path
from datetime import date
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from risk_engine.config import SETTINGS
from risk_engine.data.preprocess import build_dataset
from risk_engine.models.impact import compute_expected_loss_df

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = PROJECT_ROOT / "src" / "reports" / "templates"
OUTPUT_DIR = PROJECT_ROOT / SETTINGS["reports"]["output_dir"]

def generate_weekly_report():
    df, X, y, _ = build_dataset()
    df_scores = compute_expected_loss_df(df)  # maakt churn_proba + expected_loss

    # overall
    overall = {
        "avg_proba": float(df_scores["churn_proba"].mean()),
        "revenue_at_risk_month": float(
            (df_scores["churn_proba"] * df_scores["MonthlyCharges"]).sum()
        ),
        "expected_loss_total": float(df_scores["expected_loss_total"].sum()),
    }

    # segments
    seg = (
        df_scores.groupby(["Contract", "tenure_bucket"], observed=False)
        .agg(
            n_customers=("churn_proba", "size"),
            avg_proba=("churn_proba", "mean"),
            total_expected_loss=("expected_loss_total", "sum"),
        )
        .reset_index()
        .sort_values("total_expected_loss", ascending=False)
        .head(10)
    )

    # top customers
    top_customers = (
        df_scores.sort_values("expected_loss_total", ascending=False)
        .head(10)[["MonthlyCharges", "churn_proba", "expected_loss_total"]]
        .rename(columns={"expected_loss_total": "expected_loss"})
    )

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("weekly_template.html")

    context = {
        "week_label": date.today().isoformat(),
        "overall": overall,
        "segments": seg.to_dict(orient="records"),
        "top_customers": top_customers.to_dict(orient="records"),
    }

    html_str = template.render(**context)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"report_{date.today().strftime('%Y%m%d')}.pdf"

    HTML(string=html_str).write_pdf(str(out_path))
    print(f"Weekly report generated at {out_path}")


if __name__ == "__main__":
    generate_weekly_report()
