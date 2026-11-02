"""Reusable feature-engineering functions for the Formula 1 project."""

import pandas as pd


def add_driver_age(dataframe):
    """Return a copy of the data with driver age at race date."""
    data = dataframe.copy()

    data["date"] = pd.to_datetime(data["date"])
    data["dob"] = pd.to_datetime(data["dob"])

    data["driver_age"] = (
        (data["date"] - data["dob"]).dt.days / 365.25
    ).astype(int)

    return data
