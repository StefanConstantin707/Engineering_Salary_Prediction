"""
Path configuration for the project.
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Experiment directories
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
SUBMISSIONS_DIR = RESULTS_DIR / "submissions"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Notebook directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Source directory
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if they don't exist
for dir_path in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    CONFIGS_DIR, RUNS_DIR, SUBMISSIONS_DIR, MODELS_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"

# Default model path
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Ensure .gitkeep files in empty directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, SUBMISSIONS_DIR]:
    gitkeep_path = dir_path / ".gitkeep"
    if not gitkeep_path.exists():
        gitkeep_path.touch()


def get_latest_model(model_dir=MODELS_DIR):
    """Get the path to the latest saved model."""
    model_files = list(model_dir.glob("*.pkl"))
    if not model_files:
        return None
    # Sort by modification time
    return max(model_files, key=lambda p: p.stat().st_mtime)


def get_latest_submission(submission_dir=SUBMISSIONS_DIR):
    """Get the path to the latest submission file."""
    submission_files = list(submission_dir.glob("*.csv"))
    if not submission_files:
        return None
    # Sort by modification time
    return max(submission_files, key=lambda p: p.stat().st_mtime)