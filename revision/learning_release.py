"""Publish the saved-trajectory plots and their numerical records."""
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FILES = {
    "focus/learning_plots/learning_main.png": "iterative-wsp-training-curves.png",
    "focus/learning_plots/learning_tasks.png": "iterative-wsp-task-training-curves.png",
    "focus/learning_plots/learning_proposals.png": "iterative-wsp-proposal-curves.png",
    "focus/learning_plots/selected_rounds.csv": "iterative-wsp-learning-curves.csv",
    "focus/learning_plots/round_summary.csv": "iterative-wsp-learning-summary.csv",
    "focus/learning_plots/proposal_rounds.csv": "iterative-wsp-proposal-curves.csv",
    "focus/main_visuals/prediction_curves.png": "function-prediction-curves.png",
    "focus/main_visuals/prediction_curves.csv": "function-prediction-curves.csv",
}
for source, name in FILES.items():
    shutil.copyfile(ROOT / source, OUTPUT / name)
print(f"Published {len(FILES)} plot and data files.")
