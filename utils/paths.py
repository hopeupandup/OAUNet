"""Repository-relative path resolution."""

from pathlib import Path

OAUNET_DIR = Path(__file__).resolve().parent.parent

PROJECT_ROOT = OAUNET_DIR

# Put the CIKM2017 train_inputs/train_outputs/train_labels .npy files here; see README section 4.
SOURCE_DATA_DIR = OAUNET_DIR / 'data_raw'

TEMPORAL_DATA_DIR = OAUNET_DIR / 'data_temporal'

RESULT_DIR = OAUNET_DIR / 'results'

DATA_DIR = TEMPORAL_DATA_DIR
CHECKPOINTS_DIR = RESULT_DIR / 'checkpoints'
METRICS_DIR = RESULT_DIR / 'metrics'
FIGURES_DIR = RESULT_DIR / 'figures'
LOGS_DIR = RESULT_DIR / 'logs'


def find_temporal_split(prefer=None):
    if prefer:
        name = prefer if prefer.endswith('.json') else f'{prefer}_split.json'
        cand = TEMPORAL_DATA_DIR / name
        return cand if cand.exists() else None

    fixed = TEMPORAL_DATA_DIR / 'temporal_split.json'
    if fixed.exists():
        return fixed

    cands = sorted(TEMPORAL_DATA_DIR.glob('temporal_*_split.json'))
    if len(cands) == 1:
        return cands[0]
    return None


def resolve_config_path(rel_path):
    if rel_path is None:
        return None
    p = Path(rel_path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())
