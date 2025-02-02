"""
This is an application that will allow us to easily run all of our experiments.
"""

import os
import shutil
import typing as t
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache, partial

import typer
import yaml
import joblib
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

import data
import model
import metrics

app = typer.Typer()


@app.command()
def train(config_file: str):
    estimator_config = _load_config(config_file, "estimator")
    split = "train"
    X, y = _get_dataset(_load_config(config_file, "data"),
                        splits=[split])[split]
    estimator = model.build_estimator(estimator_config)
    if 'catboost' in estimator.named_steps.keys():
        estimator = estimator.named_steps['catboost']
        estimator.fit(
            X, y, cat_features=data.get_categorical_column_names('modelcols_1.json'))
    else:
        estimator.fit(X, y)
    output_dir = _load_config(config_file, "export")["output_dir"]
    _save_versioned_estimator(estimator, estimator_config, output_dir)


def _get_dataset(data_config, splits):
    filepath = data_config["filepath"]
    reader = partial(pd.read_csv, filepath_or_buffer=filepath)
    return data.get_dataset(reader=reader, splits=splits)


def _save_versioned_estimator(
    estimator: BaseEstimator, config: model.EstimatorConfig, output_dir: str
):
    version = datetime.now(timezone.utc).strftime("%Y-%m-%d %H-%M")
    model_dir = os.path.join(output_dir, version)
    os.makedirs(model_dir, exist_ok=True)
    try:
        joblib.dump(estimator, os.path.join(model_dir, "model.joblib"))
        _save_yaml(config, os.path.join(model_dir, "params.yml"))
    except Exception as e:
        typer.echo(f"Coudln't serialize model due to error {e}")
        shutil.rmtree(model_dir)


class CustomCrossValidation:
    @classmethod
    def split(cls,
              X: pd.DataFrame,
              y: np.ndarray = None,
              groups: np.ndarray = None):
        """Returns to a grouped time series split generator."""
        assert len(X) == len(groups),  (
            "Length of the predictors is not"
            "matching with the groups.")
        # The min max index must be sorted in the range
        for group_idx in range(groups.min(), groups.max()):
            training_group = group_idx
            # Gets the next group right after
            # the training as test
            test_group = group_idx + 2
            training_indices = np.where(groups == training_group)[0]
            test_indices = np.where(groups == test_group)[0]
            if len(test_indices) > 0:
                # Yielding to training and testing indices
                # for cross-validation generator
                yield training_indices, test_indices


def _load_dataset(data_config):
    filepath = data_config["filepath"]
    return pd.read_csv(filepath, index_col=0, encoding='utf-8').reset_index()


def define_x_y_groups(df):
    map_dict = {}
    for i, camp in enumerate(df.AÑO_CAMPAÑA.unique()):
        map_dict[camp] = i
    groups = df.AÑO_CAMPAÑA.map(map_dict)
    y = df.CANTPED
    X = df.drop(['CANTPED', "AÑO_CAMPAÑA"], axis=1)
    print(X)
    return X, y, groups


@ app.command()
def find_hyperparams(
    config_file: str,
):
    search_config = _load_config(config_file, "search")
    param_grid = search_config["grid"]
    n_jobs = search_config["jobs"]
    metric = _load_config(config_file, "metrics")[0]
    estimator_config = _load_config(config_file, "estimator")
    estimator = model.build_estimator(estimator_config)
    scoring = metrics.get_scoring_function(metric["name"], **metric["params"])

    split = "train"
    df = _load_dataset(_load_config(config_file, "data"))

    X, y, groups = define_x_y_groups(df)
    custom_splitter = CustomCrossValidation.split(
        X=X,
        y=y,
        groups=groups)

    gs = GridSearchCV(
        estimator,
        _param_grid_to_sklearn_format(param_grid),
        n_jobs=n_jobs,
        scoring=scoring,
        verbose=3,
        cv=custom_splitter)

    gs.fit(X, y)
    estimator_config = _param_grid_to_custom_format(gs.best_params_)
    estimator = gs.best_estimator_
    output_dir = _load_config(config_file, "export")["output_dir"]
    _save_versioned_estimator(estimator, estimator_config, output_dir)


def _param_grid_to_sklearn_format(param_grid):
    result = {
        f"{spec['name']}__{pname}": pvalues
        for spec in param_grid
        for pname, pvalues in spec["params"].items()
    }
    return result


def _param_grid_to_custom_format(param_grid):
    grid = {}
    for name, values in param_grid.items():
        estimator_name, param_name = name.split("__", maxsplit=1)
        if estimator_name not in grid:
            grid[estimator_name] = {}
        grid[estimator_name][param_name] = values
    result = grid
    result = [{"name": name, "hparams": params}
              for name, params in grid.items()]
    return result


@ app.command()
def eval(
    config_file: str,
    model_version: str,
    splits: t.List[str] = ["test"],


):
    output_dir = _load_config(config_file, "export")["output_dir"]
    saved_model = os.path.join(output_dir, model_version, "model.joblib")
    estimator = joblib.load(saved_model)
    dataset = _get_dataset(_load_config(config_file, "data"), splits=splits)
    report = defaultdict(list)
    all_metrics = _load_config(config_file, "metrics")
    for name, (X, y) in dataset.items():
        y_pred = estimator.predict(X)
        for m in all_metrics:
            metric_name, params = m["name"], m["params"]
            fn = metrics.get_metric_function(metric_name, **params)
            value = float(fn(y, y_pred))
            report[metric_name].append({"split": name, "value": value})
    reports_dir = _load_config(config_file, "reports")["dir"]
    os.makedirs(reports_dir, exist_ok=True)
    _save_yaml(
        dict(report),
        os.path.join(reports_dir, f"{model_version}.yml"),
    )


def _load_config(filepath: str, key: str):
    content = _load_yaml(filepath)
    config = content[key]
    return config


@ lru_cache(None)
def _load_yaml(filepath: str) -> t.Dict[str, t.Any]:
    with open(filepath, "r") as f:
        content = yaml.load(f)
    return content


def _save_yaml(content: t.Dict[str, t.Any], filepath: str):
    with open(filepath, "w") as f:
        yaml.dump(content, f)


if __name__ == "__main__":
    app()
