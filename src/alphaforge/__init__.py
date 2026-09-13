"""AlphaForge research data pipeline."""

from alphaforge.dataset import DatasetConfig, ResearchDataset, build_dataset
from alphaforge.baselines import BaselineReport, evaluate_baselines
from alphaforge.classical import evaluate_classical_models
from alphaforge.splits import ChronologicalSplitConfig, chronological_split
from alphaforge.walk_forward import WalkForwardConfig, evaluate_walk_forward
from alphaforge.robustness import RobustnessEvidence, robustness_verdict

__all__ = [
    "BaselineReport",
    "ChronologicalSplitConfig",
    "DatasetConfig",
    "ResearchDataset",
    "RobustnessEvidence",
    "WalkForwardConfig",
    "build_dataset",
    "chronological_split",
    "evaluate_baselines",
    "evaluate_classical_models",
    "evaluate_walk_forward",
    "robustness_verdict",
]
