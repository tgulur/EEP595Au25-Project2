"""
main_experiment.py - Main script for training and evaluating CAN IDS system
"""

import os
import sys
import yaml
import numpy as np
import argparse
from typing import Optional, Tuple
import logging
import logging.config
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime

# Use package-style imports from the `src` package (repo root is on PYTHONPATH)
from src.dataset_loader import CANDatasetLoader

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        }
    },
    "root": {"handlers": ["console"], "level": "INFO"},
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class CANIDSExperiment:
    """Main experiment class for CAN IDS"""
    
    def __init__(self, config_path: str = "config/config.yaml", seed: Optional[int] = None):
        """
        Initialize experiment
        
        Args:
            config_path: Path to configuration file
        """
        # Load config: keep raw dict for backward compatibility and provide
        # a dot-access namespace at `self.cfg` for new code to use.
        raw, ns = self.load_config(config_path)
        self.config = raw
        self.cfg = ns
        # Optional seed override (helps reproducibility)
        if seed is not None:
            try:
                np.random.seed(int(seed))
            except Exception:
                pass
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(self.config['output']['results_path']) / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Best-effort: configure CUDA paths from any in-project venv
        try:
            from src.configuration import configure_cuda_paths

            configure_cuda_paths()
        except Exception:
            pass

        # Load and snapshot config using centralized helper
        try:
            from src.configuration import load_and_snapshot_config

            # load_and_snapshot_config already writes metadata.json into results_dir
            raw, ns, resolved_results_dir, ts = load_and_snapshot_config(config_path, results_base=self.config.get('output', {}).get('results_path'))
            # override local variables with resolved values just in case
            self.config = raw
            self.cfg = ns
            self.results_dir = Path(resolved_results_dir)
            self.timestamp = ts
        except Exception:
            logging.getLogger(__name__).warning("Failed to run consolidated load_and_snapshot_config; falling back to previous snapshot approach")
            try:
                from src.configuration import snapshot_config

                extra = {}
                if isinstance(self.config, dict):
                    seed_val = self.config.get('data', {}).get('random_seed')
                    if seed_val is not None:
                        extra['random_seed'] = seed_val

                snapshot_config(self.config, str(self.results_dir), self.timestamp, extra=extra)
            except Exception:
                logging.getLogger(__name__).warning("Failed to write metadata snapshot")

        # Instantiate evaluator lazily to avoid heavy imports at module import time
        from src.evaluation_metrics import IDSEvaluator

        self.evaluator = IDSEvaluator(save_dir=str(self.results_dir))
        self.results = {}
        
        logger.info(f"Experiment initialized. Results will be saved to {self.results_dir}")
    
    def load_config(self, config_path: str) -> Tuple[dict, object]:
        """Load configuration from YAML and return (raw_dict, namespace)

        Uses `src.configuration.load_config` for a small, dependency-free
        dot-access wrapper while preserving the original dict layout.
        """
        from src.configuration import load_config as _load_config

        raw, ns = _load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        return raw, ns
    
    def load_and_prepare_data(self):
        """Load and prepare datasets"""
        logger.info("Loading datasets...")
        
        loader = CANDatasetLoader(self.config['data']['raw_path'])
        
        # Load voltage dataset
        voltage_df = loader.load_canmap_voltage_dataset(
            self.config['data'].get('canmap_path')
        )
        
        # Load CAN message dataset
        can_df = loader.load_road_dataset(
            self.config['data'].get('road_path')
        )
        
        # Preprocess voltage data
        logger.info("Preprocessing voltage data...")
        X_voltage, y_voltage = loader.preprocess_voltage_data(voltage_df)
        
        # Preprocess CAN message data
        logger.info("Preprocessing CAN message data...")
        sequence_length = self.config['deep_learning']['sequence_length']
        X_can, y_can = loader.preprocess_can_data(can_df, sequence_length=sequence_length)
        
        # Split data
        voltage_splits = loader.split_data(
            X_voltage, y_voltage,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            random_seed=self.config['data']['random_seed']
        )
        
        can_splits = loader.split_data(
            X_can, y_can,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            random_seed=self.config['data']['random_seed']
        )
        
        logger.info("Data preparation complete")
        
        return {
            'voltage': voltage_splits,
            'can': can_splits,
            'voltage_df': voltage_df,
            'can_df': can_df
        }
    
    def train_voltage_model(self, data: dict):
        """Train voltage fingerprinting model (delegates to trainer)."""
        from src.trainers.voltage import train_voltage_model

        fingerprinter, result = train_voltage_model(self.config, data, self.evaluator, str(self.results_dir))
        self.results['voltage'] = result
        return fingerprinter
    
    def train_baseline_models(self, data: dict):
        """Train baseline IDS models for comparison (delegates to trainer)."""
        from src.trainers.baselines import train_baseline_models

        baseline_results = train_baseline_models(self.config, data, self.evaluator, str(self.results_dir))
        self.results['baselines'] = baseline_results
        return baseline_results
    
    def train_deep_learning_models(self, data: dict):
        """Train deep learning models (delegates to trainer)."""
        from src.trainers.deep_learning import train_deep_learning_models

        dl_results = train_deep_learning_models(self.config, data, self.evaluator, str(self.results_dir))
        self.results['deep_learning'] = dl_results
        return dl_results
    
    def train_fusion_layer(self, voltage_model, dl_models, data: dict):
        """Train fusion layer (delegates to `src.trainers.fusion`)."""
        from src.trainers.fusion import train_fusion_layer

        fusion_model, fusion_result = train_fusion_layer(
            self.config,
            self.results,
            data,
            self.evaluator,
            str(self.results_dir)
        )

        self.results['fusion'] = fusion_result
        return fusion_model
    
    def compare_all_models(self):
        """Generate comprehensive comparison visualizations (delegates to `src.evaluation`)."""
        from src.evaluation import compare_and_visualize

        compare_and_visualize(self.evaluator, self.results, str(self.results_dir))

    def run_ablation_study(self):
        """Run ablation study (delegates to `src.evaluation`)."""
        from src.evaluation import run_ablation_study

        run_ablation_study(self.evaluator, self.results, str(self.results_dir))
    
    def run_experiment(self):
        """Run full experiment"""
        logger.info("\n" + "="*60)
        logger.info("Starting CAN IDS Experiment")
        logger.info("="*60)
        
        try:
            # Load data
            data = self.load_and_prepare_data()
            
            # Train voltage model
            voltage_model = self.train_voltage_model(data)
            
            # Train baseline models
            baseline_models = self.train_baseline_models(data)
            
            # Train DL models
            dl_models = self.train_deep_learning_models(data)
            
            # Train fusion layer
            fusion_model = self.train_fusion_layer(voltage_model, dl_models, data)
            
            # Compare models (including baselines)
            self.compare_all_models()
            
            # Run ablation study
            self.run_ablation_study()
            
            logger.info("\n" + "="*60)
            logger.info("Experiment Complete!")
            logger.info(f"Results saved to: {self.results_dir}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


def main():
    # Delegate CLI and orchestration to `src.runner` to keep this file
    # as a thin shim and make refactors easier.
    from src.runner import run as runner_run

    # Pass the experiment class so `src.runner` does not need to import this
    # module directly (avoids circular import pitfalls during refactors).
    raise SystemExit(runner_run(CANIDSExperiment))


if __name__ == "__main__":
    main()
