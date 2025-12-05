from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "src" / "config" / "settings.yaml"

with open(SETTINGS_PATH) as f:
    SETTINGS = yaml.safe_load(f)
