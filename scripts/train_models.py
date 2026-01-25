"""CLI wrapper to train and persist models (calls training service)."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from recsys_app.services.training import train_and_persist_models

if __name__ == '__main__':
    print("Training models...")
    res = train_and_persist_models(simulate_interactions=True)
    print("Training done. Saved files:", res.get('models'))
