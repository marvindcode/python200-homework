import numpy as np
import pandas as pd

from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")    


@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow
def pipeline_flow(arr):
    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)
    return summary

if __name__ == "__main__":
    result = pipeline_flow(arr)




