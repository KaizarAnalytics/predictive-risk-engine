from src.reports.weekly_report import generate_weekly_report
from pathlib import Path

def test_generate_weekly_report(tmp_path, monkeypatch):
    # override output dir
    from src.config import SETTINGS
    SETTINGS["reports"]["output_dir"] = str(tmp_path)

    generate_weekly_report()
    pdfs = list(tmp_path.glob("report_*.pdf"))
    assert len(pdfs) == 1
