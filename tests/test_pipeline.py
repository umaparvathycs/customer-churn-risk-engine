from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_generation import FEATURES, TARGET, generate_dataset


def test_dataset_is_reproducible_and_well_formed():
    first = generate_dataset(300, 42)
    second = generate_dataset(300, 42)
    pd.testing.assert_frame_equal(first, second)
    assert set(FEATURES + [TARGET]).issubset(first.columns)
    assert first[TARGET].nunique() == 2


def test_model_artifacts_exist_after_training():
    assert (ROOT / "models" / "churn_pipeline.joblib").exists()
    assert (ROOT / "models" / "metrics.json").exists()
