from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
EVAL_DIR = BASE_DIR / "evaluation"

# Logistic Regression evaluation files
LR_METRICS_PATH = EVAL_DIR / "lr_3class_metrics.json"
LR_CM_PATH = EVAL_DIR / "lr_3class_confusion_matrix.csv"
LR_REPORT_PATH = EVAL_DIR / "lr_3class_classification_report.csv"

# Logistic Regression model file (lipsea!)
LR_MODEL_PATH = BASE_DIR / "models" / "lr_3class_balanced.joblib"

# Manual validation files
MANUAL_SAMPLE_PATH = EVAL_DIR / "reddit_manual_validation_sample.csv"
MANUAL_METRICS_PATH = EVAL_DIR / "reddit_manual_validation_metrics.json"
MANUAL_CM_PATH = EVAL_DIR / "reddit_manual_validation_confusion_matrix.csv"
MANUAL_REPORT_PATH = EVAL_DIR / "reddit_manual_validation_report.csv"
