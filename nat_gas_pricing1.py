"""
Natural Gas Price Estimator
============================
Loads the monthly natural gas price snapshots (month-end market price, 31-Oct-2020
through 30-Sep-2024), fits a trend + seasonal model, and exposes a function that
returns a price estimate for ANY date -- past (via interpolation against the
historical model) or up to one year beyond the last data point (via extrapolation
of the fitted trend + seasonal cycle).

Model choice
------------
Natural gas storage economics are driven by:
  1. A long-term price TREND (supply/demand growth, inflation, LNG export
     capacity additions, etc.)
  2. A repeating SEASONAL cycle within the calendar year (prices typically rise
     into the winter heating season Nov-Feb and soften in the shoulder/summer
     injection months, reflecting the classic contango/backwardation pattern
     described for Henry Hub storage economics).

We fit:   price(t) = a + b*t + c*sin(2*pi*t/365.25) + d*cos(2*pi*t/365.25)
where t = days since the first observation. The sin/cos pair lets the model
learn both the amplitude AND phase of the seasonal cycle (i.e., it will find
that prices peak in winter without us having to hard-code which month).

For dates that fall INSIDE the historical data range we use linear
interpolation between the two bracketing month-end snapshots (this exactly
reproduces the observed data and is more accurate locally than the smooth
regression fit). For dates OUTSIDE the range (future estimates, or up to one
year past the last snapshot) we use the fitted trend + seasonal model.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CSV_PATH = "Nat_Gas.csv"


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df["Prices"] = df["Prices"].astype(float)
    df = df.sort_values("Dates").reset_index(drop=True)
    return df


def fit_model(df: pd.DataFrame):
    """Fit price = a + b*t + c*sin(2*pi*t/365.25) + d*cos(2*pi*t/365.25)."""
    t0 = df["Dates"].iloc[0]
    t = (df["Dates"] - t0).dt.days.values.astype(float)
    y = df["Prices"].values.astype(float)

    period = 365.25
    X = np.column_stack([
        np.ones_like(t),
        t,
        np.sin(2 * np.pi * t / period),
        np.cos(2 * np.pi * t / period),
    ])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs, t0


def model_price(date, coeffs, t0):
    t = (pd.Timestamp(date) - t0).days
    a, b, c, d = coeffs
    period = 365.25
    return a + b * t + c * np.sin(2 * np.pi * t / period) + d * np.cos(2 * np.pi * t / period)


class NatGasPriceEstimator:
    def __init__(self, csv_path: str = CSV_PATH):
        self.df = load_data(csv_path)
        self.coeffs, self.t0 = fit_model(self.df)
        self.min_date = self.df["Dates"].min()
        self.max_date = self.df["Dates"].max()
        self.max_extrapolation_date = self.max_date + pd.DateOffset(years=1)

    def estimate(self, date) -> float:
        """
        Return an estimated natural gas purchase price for `date`
        (str 'YYYY-MM-DD', 'MM/DD/YY', or datetime-like).

        - Within [min_date, max_date]: linear interpolation between the two
          nearest month-end observations (reproduces historical data exactly
          on observed dates).
        - Beyond max_date (up to +1 year): trend + seasonal model
          extrapolation.
        - Before min_date: trend + seasonal model (backcast), with a caveat.
        """
        date = pd.Timestamp(date)

        if date > self.max_extrapolation_date:
            raise ValueError(
                f"Date {date.date()} is more than 1 year beyond the last "
                f"observation ({self.max_date.date()}). Extrapolation beyond "
                f"that horizon is not supported."
            )

        if self.min_date <= date <= self.max_date:
            # Interpolate against actual observed snapshots.
            x = self.df["Dates"].values.astype("datetime64[D]").astype(float)
            y = self.df["Prices"].values
            xq = np.array(date.to_datetime64()).astype("datetime64[D]").astype(float)
            return float(np.interp(xq, x, y))

        # Outside observed range -> use fitted trend + seasonal model.
        return float(model_price(date, self.coeffs, self.t0))

    def estimate_range(self, start_date, end_date, freq="D") -> pd.Series:
        dates = pd.date_range(start_date, end_date, freq=freq)
        prices = [self.estimate(d) for d in dates]
        return pd.Series(prices, index=dates, name="EstimatedPrice")


if __name__ == "__main__":
    est = NatGasPriceEstimator()

    print("Data range:", est.min_date.date(), "to", est.max_date.date())
    print("Extrapolation supported through:", est.max_extrapolation_date.date())
    print()

    test_dates = [
        "2021-03-15",   # interpolated, in-sample
        "2023-07-04",   # interpolated, in-sample
        "2024-09-30",   # last observed point
        "2024-12-31",   # extrapolated, winter peak expected
        "2025-06-30",   # extrapolated, summer trough expected
        "2025-09-30",   # extrapolated, 1 year beyond last data point
    ]
    for d in test_dates:
        print(f"{d}: ${est.estimate(d):.2f}")
