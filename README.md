# Kaizar Risk Engine (Open Core)

A modular, production-style churn risk engine focused on clean ML architecture, explainability, and automated reporting. This open-core version contains the full modeling pipeline, while the production API and dashboard remain proprietary.

Full case description: [kaizar.nl/posts/risk-engine.html](https://kaizar.nl/posts/risk-engine.html)

## Example Results

Using the Telco Customer Churn dataset with the default configuration:

| Metric | Value |
|--------|-------|
| Top 20% risk segment | Captures ~67% of all churn |
| Targeting efficiency | ~3x more effective than random |
| Explainability | Per-customer SHAP breakdowns |

These numbers are indicative for the demo setup and will vary with your own data.

## Features

- **Data preprocessing & feature engineering** - Automated pipeline for raw data
- **XGBoost churn model** - Training, evaluation, and model artifact export
- **SHAP explainability** - Global feature importance and local explanations
- **Automated PDF reporting** - Weekly risk reports using Jinja2 + WeasyPrint
- **Modular codebase** - Production-ready structure
- **Jupyter notebooks** - Step-by-step walkthrough of the full pipeline

## Repository Structure

```
src/
  data/           # Loading & preprocessing logic
  models/         # Training, evaluation, SHAP analysis
  reports/        # PDF report generation
  config/         # YAML configs for thresholds & paths
notebooks/        # Exploratory notebooks (01-05)
tests/            # Unit tests
```

The production API and Streamlit dashboard are intentionally **not included**.

## Installation

```bash
git clone https://github.com/KaizarAnalytics/predictive-risk-engine.git
cd predictive-risk-engine
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

```bash
# Train the model
risk-engine train

# Score customers and show top risks
risk-engine predict --top 20

# Show model information
risk-engine info
```

### Train the model (alternative)

```bash
python -m src.models.train_model
```

### Generate predictions

```python
from src.models.train_model import get_model
from src.data.preprocess import load_raw
from src.data.feature_engineering import apply_feature_engineering

model = get_model()
df, _ = load_raw()
df = apply_feature_engineering(df)

# Get churn probabilities
proba = model.predict_proba(df)[:, 1]
```

### Generate weekly report

```bash
python -m src.reports.weekly_report
```

## Notebooks

Interactive demonstrations in `notebooks/`:

1. **01_exploration.ipynb** - Data exploration and baseline analysis
2. **02_feature_engineering.ipynb** - Feature creation and transformation
3. **03_model_training.ipynb** - XGBoost model training with cross-validation
4. **04_shap_analysis.ipynb** - SHAP-based feature importance and explanations
5. **05_impact_calculation.ipynb** - Business impact and expected loss calculations

## Testing

```bash
pip install pytest
pytest -v
```

## Data Notice

This project uses the **Telco Customer Churn** dataset from Kaggle:
[kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)

The dataset is **not included** in this repository. Users must download it directly from Kaggle (handled automatically via `kagglehub`).

## License

Source code is proprietary to Kaizar. Modeling components are provided for learning and experimentation. Commercial deployment requires a license agreement.

## About

Developed by [Kaizar](https://kaizar.nl) - modeling systems, risk intelligence, and applied machine learning.
