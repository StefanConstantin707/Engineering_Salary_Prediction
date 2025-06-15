import os
import json
import pickle
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List


class RunTracker:
    """
    Track and save machine learning experiment runs with detailed diagnostics.
    """

    def __init__(self, base_dir: str = "runs", run_name: str = "experiment"):
        """
        Initialize the RunTracker.

        Args:
            base_dir: Base directory for saving runs
            run_name: Name of the current experiment run
        """
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.base_dir.mkdir(exist_ok=True)

        # Get the next run index
        self.run_index = self._get_next_run_index()

        # Initialize run data
        self.run_data = {
            "run_name": run_name,
            "run_index": self.run_index,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "best_score": None,
            "best_params": None,
            "top_results": [],
            "cv_results": None,
            "feature_importance": None,
            "outlier_info": None,
            "model_info": {},
            "diagnostics": {},
            "iteration_history": []
        }

    def _get_next_run_index(self) -> int:
        """Get the next available run index."""
        existing_files = list(self.base_dir.glob("*.json"))
        if not existing_files:
            return 1

        indices = []
        for file in existing_files:
            parts = file.stem.split("_")
            if parts[-1].startswith("run"):
                try:
                    idx = int(parts[-1][3:])
                    indices.append(idx)
                except ValueError:
                    continue

        return max(indices, default=0) + 1

    def log_iteration(self, iteration: int, params: Dict[str, Any], score: float,
                      std: float, duration: float):
        """
        Log a single optimization iteration.

        Args:
            iteration: Iteration number
            params: Parameters used in this iteration
            score: CV score achieved
            std: Standard deviation of CV score
            duration: Time taken for this iteration
        """
        iteration_data = {
            "iteration": iteration,
            "params": params,
            "score": score,
            "std": std,
            "duration": duration,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.run_data["iteration_history"].append(iteration_data)

        # Print iteration info
        print(f"\nIteration {iteration:3d} | Score: {score:.4f} ± {std:.4f} | "
              f"Duration: {duration:.2f}s")
        print(f"Parameters: {self._format_params(params)}")

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for display."""
        formatted = []
        for key, value in params.items():
            short_key = key.split("__")[-1]
            if isinstance(value, float):
                formatted.append(f"{short_key}={value:.6f}")
            else:
                formatted.append(f"{short_key}={value}")
        return " | ".join(formatted)

    def save_optimization_results(self, bayes_search, X_train, y_train,
                                  outlier_detector=None, feature_selector_info=None):
        """
        Save the results from BayesSearchCV optimization.

        Args:
            bayes_search: Fitted BayesSearchCV object
            X_train: Training features (for shape info)
            y_train: Training labels (for distribution info)
            outlier_detector: Optional outlier detector object
            feature_selector_info: Optional dict with feature selection info
        """
        # Make sure bayes_search is fitted
        if not hasattr(bayes_search, 'cv_results_'):
            raise ValueError("BayesSearchCV must be fitted before saving results")

        # Extract and save best score and params
        self.run_data["best_score"] = float(bayes_search.best_score_)
        self.run_data["best_params"] = bayes_search.best_params_

        # Get top 5 results
        cv_results_df = pd.DataFrame(bayes_search.cv_results_)
        top_5_indices = cv_results_df.nlargest(5, 'mean_test_score').index

        top_results = []
        for idx in top_5_indices:
            # Get parameters for this iteration
            # cv_results_['params'] is a list of dicts
            params_dict = bayes_search.cv_results_['params'][idx]

            result = {
                "rank": int(cv_results_df.loc[idx, 'rank_test_score']),
                "score": float(cv_results_df.loc[idx, 'mean_test_score']),
                "std": float(cv_results_df.loc[idx, 'std_test_score']),
                "params": params_dict,  # This is already a dict
                "fit_time": float(cv_results_df.loc[idx, 'mean_fit_time']),
                "score_time": float(cv_results_df.loc[idx, 'mean_score_time'])
            }
            top_results.append(result)

        self.run_data["top_results"] = top_results

        # Save full CV results summary
        self.run_data["cv_results_summary"] = {
            "n_iterations": len(cv_results_df),
            "total_fit_time": float(cv_results_df['mean_fit_time'].sum()),
            "score_distribution": {
                "min": float(cv_results_df['mean_test_score'].min()),
                "max": float(cv_results_df['mean_test_score'].max()),
                "mean": float(cv_results_df['mean_test_score'].mean()),
                "std": float(cv_results_df['mean_test_score'].std())
            }
        }

        # Save outlier information if provided
        if outlier_detector is not None:
            try:
                self.run_data["outlier_info"] = {
                    "method": outlier_detector.method,
                    "contamination": outlier_detector.contamination,
                    "n_outliers": int(np.sum(outlier_detector.outlier_mask_)),
                    "outlier_percentage": float(np.sum(outlier_detector.outlier_mask_) / len(outlier_detector.outlier_mask_) * 100)
                }
            except Exception as e:
                print(f"Warning: Could not save outlier info: {e}")
                self.run_data["outlier_info"] = None

        # Save feature selection info if provided
        if feature_selector_info is not None:
            self.run_data["feature_selection_info"] = feature_selector_info

        # Save data shape and distribution info
        self.run_data["data_info"] = {
            "train_shape": list(X_train.shape),  # Convert to list for JSON serialization
            "target_distribution": {}
        }

        # Handle target distribution safely
        try:
            if hasattr(y_train, 'value_counts'):
                # Convert to regular dict with int keys (for JSON serialization)
                value_counts = y_train.value_counts()
                self.run_data["data_info"]["target_distribution"] = {
                    int(k): int(v) for k, v in value_counts.to_dict().items()
                }
        except Exception as e:
            print(f"Warning: Could not save target distribution: {e}")

        # Mark end time
        self.run_data["end_time"] = datetime.datetime.now().isoformat()

        print(f"Optimization results saved successfully.")
        print(f"Best score: {self.run_data['best_score']:.4f}")
        print(f"Total iterations: {self.run_data['cv_results_summary']['n_iterations']}")

    def save_feature_importance(self, importance_df: pd.DataFrame):
        """Save feature importance information."""
        if importance_df is not None and len(importance_df) > 0:
            self.run_data["feature_importance"] = importance_df.to_dict('records')

    def add_diagnostic(self, key: str, value: Any):
        """Add a diagnostic metric."""
        self.run_data["diagnostics"][key] = value

    def save_to_file(self, include_model: bool = False, model=None):
        """
        Save the run data to a JSON file and optionally save the model.

        Args:
            include_model: Whether to save the model as a pickle file
            model: The model object to save (if include_model is True)
        """
        # Create filename
        score_str = f"{self.run_data['best_score']:.4f}" if self.run_data['best_score'] else "incomplete"
        filename = f"{self.run_name}_score{score_str}_run{self.run_index:03d}"

        # Save JSON data
        json_path = self.base_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(self.run_data, f, indent=2, default=str)

        print(f"\nRun data saved to: {json_path}")

        # Save model if requested
        if include_model and model is not None:
            model_path = self.base_dir / f"{filename}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to: {model_path}")

            self.run_data["model_info"]["model_path"] = str(model_path)

        # Save a summary CSV of all runs
        self._update_summary_csv()

        return json_path

    def _update_summary_csv(self):
        """Update or create a summary CSV of all runs."""
        summary_path = self.base_dir / "runs_summary.csv"

        # Collect all run data
        runs = []
        for json_file in self.base_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                run_summary = {
                    "run_name": data.get("run_name"),
                    "run_index": data.get("run_index"),
                    "best_score": data.get("best_score"),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "n_iterations": len(data.get("iteration_history", [])),
                    "file_path": str(json_file)
                }

                # Add best parameters
                best_params = data.get("best_params", {})
                for param, value in best_params.items():
                    param_name = param.split("__")[-1]
                    run_summary[f"best_{param_name}"] = value

                runs.append(run_summary)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue

        if runs:
            summary_df = pd.DataFrame(runs)
            summary_df = summary_df.sort_values("run_index")
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary updated: {summary_path}")

    @staticmethod
    def load_run(file_path: str) -> Dict[str, Any]:
        """Load a saved run from file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def print_run_summary(self):
        """Print a summary of the current run."""
        print("\n" + "=" * 70)
        print("RUN SUMMARY")
        print("=" * 70)
        print(f"Run Name: {self.run_data['run_name']}")
        print(f"Run Index: {self.run_data['run_index']}")
        print(f"Best Score: {self.run_data['best_score']:.4f}")
        print(f"Total Iterations: {len(self.run_data['iteration_history'])}")

        if self.run_data['outlier_info']:
            print(f"Outliers Removed: {self.run_data['outlier_info']['n_outliers']} "
                  f"({self.run_data['outlier_info']['outlier_percentage']:.2f}%)")

        print("\nTop 5 Results:")
        print("-" * 70)
        for i, result in enumerate(self.run_data['top_results'], 1):
            print(f"{i}. Score: {result['score']:.4f} ± {result['std']:.4f}")
            print(f"   Parameters: {self._format_params(result['params'])}")
        print("=" * 70)


class BayesSearchCallback:
    """
    Callback for BayesSearchCV to track iterations in real-time.
    """

    def __init__(self, run_tracker: RunTracker):
        self.run_tracker = run_tracker
        self.iteration = 0
        self.start_time = None

    def __call__(self, result):
        """Called after each iteration of BayesSearchCV."""
        self.iteration += 1

        # Get the latest result
        n_calls = len(result.x_iters)
        latest_params = result.x_iters[-1]
        latest_score = -result.func_vals[-1]  # Negative because skopt minimizes

        # Estimate std (not directly available in callback)
        std = 0.0  # You might need to calculate this separately

        # Calculate duration
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
            duration = 0.0
        else:
            duration = (datetime.datetime.now() - self.start_time).total_seconds()
            self.start_time = datetime.datetime.now()

        # Create params dict (you'll need to map these based on your search space)
        # This is a simplified version - you might need to adjust based on your actual parameter names
        params_dict = {f"param_{i}": val for i, val in enumerate(latest_params)}

        # Log the iteration
        self.run_tracker.log_iteration(
            iteration=self.iteration,
            params=params_dict,
            score=latest_score,
            std=std,
            duration=duration
        )