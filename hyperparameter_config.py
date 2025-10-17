from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

from utils_module import Utils


class HyperparameterConfig:
    """Hyperparameter configuration with simple persistence helpers."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "batch_size": 8,
        "max_length": 1024,
        "lora_r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "wandb_project": "minecraft-llm",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize configuration with optional overrides."""
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self.grid_search_file = "grid-search-record.json"

    def get_config(self) -> Dict[str, Any]:
        """Return a copy of the active configuration."""
        return self.config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Apply in-place updates to the configuration."""
        self.config.update(updates)

    def get_grid_search_params(self) -> Dict[str, List[Any]]:
        """Parameters eligible for grid search sweeps."""
        return {
            "learning_rate": [1e-5, 5e-5, 1e-4],
            "batch_size": [16, 32, 64],
            "lora_r": [4, 8, 16],
            "lora_alpha": [16, 32, 64],
            "num_epochs": [3, 5],
        }

    def generate_grid_configs(self, param_grid: Optional[Dict[str, List[Any]]] = None) -> List[Dict[str, Any]]:
        """Enumerate all combinations from the provided parameter grid."""
        param_grid = param_grid or self.get_grid_search_params()
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        configs: List[Dict[str, Any]] = []
        for combination in itertools.product(*param_values):
            next_config = self.config.copy()
            for name, value in zip(param_names, combination):
                next_config[name] = value
            configs.append(next_config)

        print(f"Generated {len(configs)} grid search configurations")
        return configs

    def load_grid_search_record(self) -> Dict[str, Any]:
        """Load prior grid search progress from disk (if available)."""
        record = Utils.load_json(self.grid_search_file)
        if record is None:
            record = {
                "completed_configs": [],
                "results": [],
                "best_config": None,
                "best_score": -float("inf"),
            }
        return record

    def save_grid_search_record(self, record: Dict[str, Any]) -> None:
        """Persist grid search history to disk."""
        Utils.save_json(record, self.grid_search_file)
        print(f"Grid search record saved to {self.grid_search_file}")

    def get_next_grid_config(self, all_configs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the next untried configuration from a generated list."""
        record = self.load_grid_search_record()
        completed = set(record["completed_configs"])

        for config in all_configs:
            config_key = self._config_to_key(config)
            if config_key not in completed:
                return config
        return None

    def update_grid_search_result(self, config: Dict[str, Any], result: Dict[str, Any], score: float) -> None:
        """Record the outcome of a completed configuration."""
        record = self.load_grid_search_record()
        config_key = self._config_to_key(config)

        record["completed_configs"].append(config_key)
        record["results"].append({"config": config, "result": result, "score": score})

        if score > record["best_score"]:
            record["best_config"] = config
            record["best_score"] = score
            print(f"New best configuration found! Score: {score:.4f}")

        self.save_grid_search_record(record)

    def _config_to_key(self, config: Dict[str, Any]) -> str:
        """Create a stable string key for a configuration."""
        key_parts = []
        for param in sorted(self.get_grid_search_params().keys()):
            if param in config:
                key_parts.append(f"{param}={config[param]}")
        return "_".join(key_parts)

    def print_config(self) -> None:
        """Pretty-print the current configuration."""
        print("=" * 80)
        print("HYPERPARAMETER CONFIGURATION")
        print("=" * 80)
        for key, value in sorted(self.config.items()):
            print(f"{key:20s}: {value}")
        print("=" * 80)

    def print_grid_search_summary(self) -> None:
        """Print grid search progress and best results."""
        record = self.load_grid_search_record()

        print("=" * 80)
        print("GRID SEARCH SUMMARY")
        print("=" * 80)
        print(f"Completed configurations: {len(record['completed_configs'])}")
        print(f"Best score: {record['best_score']:.4f}")

        if record["best_config"]:
            print("\nBest configuration:")
            for key, value in sorted(record["best_config"].items()):
                if key in self.get_grid_search_params():
                    print(f"  {key:20s}: {value}")
        print("=" * 80)


def main() -> None:
    """Lightweight test harness for HyperparameterConfig."""
    config = HyperparameterConfig()
    config.print_config()
    configs = config.generate_grid_configs()
    print(f"Example configuration key: {config._config_to_key(configs[0]) if configs else 'n/a'}")


if __name__ == "__main__":
    main()

