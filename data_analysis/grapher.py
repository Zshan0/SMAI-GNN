import os
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import sys


def get_data(dir_name, column="train"):
    data = []
    dir_name = "./extracted/" + dir_name + "/"
    for file in os.listdir(dir_name):
        data.append(json.load(open(dir_name + file)))
    dfs = []
    acus = []
    for fold_data in data:
        dfs.append(
            pd.DataFrame(fold_data, columns=["epoch", "loss", "train", "test"])
        )
        acus.append(dfs[-1][column].to_list())

    averages = np.average(np.array(acus), axis=0)
    return averages


def main():
    dataset = sys.argv[1]
    datas = []
    cols = []
    column = sys.argv[2]
    for dir_name in os.listdir("extracted"):
        if dataset in dir_name:
            print(dir_name)
            cols.append(dir_name.split("-")[0])
            datas.append(get_data(dir_name, column))

    datas = [f.tolist() for f in datas]
    df = pd.DataFrame(datas)
    df = df.transpose()
    df.columns = cols

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["GIN"], mode="lines", name="GIN"))
    fig.add_trace(
        go.Scatter(x=df.index, y=df["GNN_SUM"], mode="lines", name="GNN_SUM")
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["GNN_MAX"], mode="lines", name="GNN_MAX")
    )
    fig.update_layout(
        title=f"{dataset} {column}ing error vs epochs",
        xaxis_title="Epochs",
        yaxis_title=f"{column}ing error",
        legend_title="Model name",
    )

    fig.write_image(f"{column}-{dataset}.png")


if __name__ == "__main__":
    main()
