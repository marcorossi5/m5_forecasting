import argparse
import tqdm
import glob
import os
from typing import List

from darts import TimeSeries
from darts.metrics import metrics
from darts.utils.timeseries_generation import holidays_timeseries
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pl

import models

load_dotenv("../src/.env")

LB = 28 * 4
FH = 28


def generate_covariates(series: List[TimeSeries]):
    holidays_series = holidays_timeseries(series[0].time_index, "ITA").astype(
        np.float32
    )
    return [holidays_series for _ in range(len(series))]


def load_training_data():
    data_folder = "./dataset"
    glob_str = os.path.join(data_folder, "m5_*.pkl")
    file_list = sorted(glob.glob(glob_str))

    data = {
        "fnames": [],
    }
    train = []

    for f in tqdm.tqdm(file_list, desc="loading"):
        s = TimeSeries.from_pickle(f).astype(np.float32)
        data["fnames"].append(os.path.basename(f)[3:-4])
        train.append(s[: -FH * 2])
    data["train"], data["val"] = train_test_split(train, test_size=0.1, random_state=42)
    data["train_cov"] = generate_covariates(data["train"])
    data["val_cov"] = generate_covariates(data["val"])
    return data


def load_test_data():
    data_folder = "./dataset"
    glob_str = os.path.join(data_folder, "m5_*.pkl")
    file_list = sorted(glob.glob(glob_str))

    data = {"fnames": [], "test": [], "test_labels": []}

    for f in tqdm.tqdm(file_list, desc="loading"):
        s = TimeSeries.from_pickle(f).astype(np.float32)
        data["fnames"].append(os.path.basename(f)[3:-4])
        data["test"].append(s[-FH * 5 : -FH])
        data["test_labels"].append(s[-FH:])
    data["test_cov"] = generate_covariates(data["test"])
    return data


def transform_series_to_df(
    products: List[str], series: List[TimeSeries]
) -> pd.DataFrame:
    """Postprocess the model forecasts with the reordering time series.

    :param products: the list of products codes
    :type products: List[int]
    :param forecasts: the list of model forecasts to convert
    :type forecasts: List[TimeSeries]

    :return: the dataframe containing the reordering forecasts
    :rtype: pandas dataframe
    """
    df = []
    wrap = tqdm.tqdm(zip(products, series), total=len(products))
    for product, s in wrap:
        tmp_df = s.pd_dataframe().reset_index()
        tmp_df["product"] = product
        df.append(tmp_df)

    df = pd.concat(df)
    df = df.rename(columns={"0": "quantity"})
    # Darts attaches a name to the column index, and it really bothers me.
    df.columns.rename(None, inplace=True)
    cols = ["product", "time", "quantity"]
    return df.reset_index(drop=True).sort_values(["product", "time"], ascending=False)[
        cols
    ]


def save_predictions(model_type: str, df):
    folder = f"./training_logs/preds"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    df.to_csv(f"{folder}/{model_type}_preds.csv", index=False)


def tweedie_loss(series_true, series_pred, inter_reduction=None):
    scores = []
    def log_tweedie_loss(y, y_hat, p=1.5):
        return -y_hat * np.pow(y, 1 - p) / (1 - p) + np.pow(y, 2 - p) / (2 - p)
    for y_pred, y_true in zip(series_true, series_pred):
        pred = y_pred.values().flatten()
        true = y_true.values().flatten()
        scores.append(log_tweedie_loss(true, pred))

    if inter_reduction is not None:
        return inter_reduction(scores)
    return scores


def evaluate_predictions(series_true: List[TimeSeries], series_pred: List[TimeSeries]):
    # mean_per_prod = np.array([np.mean(s.values()) for s in series_true])
    mae = metrics.mae(series_pred, series_true, inter_reduction=np.mean)
    print("MAE:", mae)
    rmse = metrics.rmse(series_pred, series_true, inter_reduction=np.mean)
    print("RMSE:", rmse)
    tweedie = tweedie_loss(series_pred, series_true, inter_reduction=np.mean)
    print("Tweedie loss:", tweedie)


def training_pipeline(model_type: str):
    print("Loading data")
    data = load_training_data()

    print("Loading model")
    model = models.create_model(model_type, LB, FH)

    trainer = models.get_trainer(model_type)

    print(f"Training length: {len(data['train'])}")
    print(f"Validation length: {len(data['val'])}")
    model.fit(
        data["train"],
        past_covariates=data["train_cov"],
        trainer=trainer,
        val_series=data["val"],
        val_past_covariates=data["val_cov"],
    )


def inference_pipeline(model_type: str, ckpt_path: str):
    print("Load testing data")
    data = load_test_data()

    # model = models.load_model(model_type, ckpt_path)

    print(f"Testing length: {len(data['test'])}")
    # preds = model.predict(
    #     model.output_chunk_length,
    #     series=data["test"],
    #     past_covariates=data["test_cov"],
    #     num_loader_workers=os.cpu_count(),
    # )

    print("Transform to predictions to csv")
    # df_preds = transform_series_to_df(data["fnames"], preds)
    df_preds = pd.read_csv(f"training_logs/preds/{model_type}_preds.csv").drop(columns="quantity")
    df_preds["time"] = pd.to_datetime(df_preds["time"])
    # df_preds["product"] = df_preds["product"].astype(object)

    # df_preds = df_preds.rename(columns={"quantity": f"{model_type}_quantity"})
    df_true = transform_series_to_df(data["fnames"], data["test_labels"])
    df_true["product"] = df_true["product"].astype(int)
    
    df = df_preds.merge(df_true, how="inner", on=["product", "time"])

    print("Saving predictions")
    save_predictions(model_type, df)

    print("Evaluate predictions")
    # evaluate_predictions(data["test_cov"], preds)


def main(args: argparse.Namespace):
    if args.mode == "training":
        training_pipeline(args.model_type)
    elif args.mode == "inference":
        inference_pipeline(args.model_type, args.ckpt_path)
    else:
        raise ValueError(f"Mode not recognized, got {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["training", "inference"], help="the program mode"
    )
    parser.add_argument(
        "-m",
        "--model_type",
        default="nhits",
        choices=["nhits", "nbeats", "tft"],
        help="the model type to be used",
    )

    parser.add_argument("-p", "--path", dest="ckpt_path", default=None, help="the model checkpoint to load")
    args = parser.parse_args()

    main(args)
