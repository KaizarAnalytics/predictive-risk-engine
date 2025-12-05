# Kaizar Risk Engine (Open Core)

A modular, production-style churn risk engine focused on clean ML architecture, explainability, and automated reporting.  
This open-core version contains the full modeling pipeline used by Kaizar, while the production API and dashboard remain proprietary.

Full case description available at:
https://kaizar.nl/posts/post.html?slug=predictive-risk

---

## Example impact (demo configuration)

Using the Telco Customer Churn dataset and the default configuration in thi>
- The top 20% highest-risk customers capture ~67% of all churn events.
- Targeting retention actions at this segment makes interventions ≈3× more >
- SHAP-based explanations provide per-customer driver breakdowns (e.g. cont>

These numbers are indicative for the demo setup and will vary when you plug>

---

## Features

- **Data preprocessing & feature engineering**
- **XGBoost churn model** with training, evaluation, and model artifact export
- **SHAP explainability** for global & local insights
- **Automated PDF reporting** using Jinja2 + WeasyPrint
- **Modular codebase** structured for production workflows
- **Kaggle & local notebooks** for experiments and reproducibility

---

## Repository Structure

```

src/
data/         # Loading & preprocessing logic
models/       # Training, evaluation, SHAP analysis, risk scoring
reports/      # Templates + PDF report generation
config/       # YAML configs for thresholds & paths
notebooks/      # Local exploratory notebooks

````

The production API and Streamlit dashboard are intentionally **not included**.

---

## Installation

```bash
pip install -r requirements.txt
````

(Optional: if packaged later)

```bash
pip install -e .
```

---

## Usage

### Train the model

```bash
python -m src.models.train_model
```

### Generate SHAP impact tables & reports

```bash
python -m src.reports.generate_weekly_report
```

### Load the model inside Python

```python
from src.models.train_model import get_model
model = get_model()
```

---

## Data Usage Notice

This project references the **blastchar/telco-customer-churn** dataset from Kaggle:

[https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)

The dataset is **not included** in this repository and cannot be redistributed due to
**“Data files © Original Authors.”**

Users must download the dataset directly from Kaggle to run the full pipeline.

---

## License

Source code © Kaizar.
Modeling components are available for learning and experimentation.
Commercial use of proprietary components (API, UI, deployment stack) is not permitted.

---

## Author

Developed by **Kaizar** — modeling systems, risk intelligence, and applied machine learning.

