"""Repository-relative locations shared by all scripts."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data' / 'hf'          # mirror of the Hugging Face dataset (data/download_data.py)
# every table and figure produced by the analyses (override with RISKATTITUDE_OUTPUT_DIR)
OUTPUT_DIR = Path(os.environ.get('RISKATTITUDE_OUTPUT_DIR', REPO_ROOT / 'outputs'))

HF_REPO = 'BowenSun1998/Benchmarking_Risk_Attitudes_of_LLMs'

FRONTIER_RELIABILITY = DATA_DIR / 'FrontierModel_reliability_N90'
EARLIER_RELIABILITY = DATA_DIR / 'EarlierModel_reliability_N90'
FRONTIER_MAIN = DATA_DIR / 'FrontierModel_main_N100'
EARLIER_MAIN = DATA_DIR / 'EarlierModel_main_N100'
HUMAN_DATA = DATA_DIR / 'RiskAttitude_HumanData'

TASKS = ('DNC', 'CTD', 'FIP')


def output_dir(module: str) -> Path:
    """Return outputs/<module>/, creating it if needed."""
    path = OUTPUT_DIR / module
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_data() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f'Dataset not found at {DATA_DIR}. Run: python data/download_data.py')
