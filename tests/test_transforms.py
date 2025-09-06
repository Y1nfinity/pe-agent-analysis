import pandas as pd
from tools.transforms import compute_holding_periods

def test_holding_periods_basic():
    df = pd.DataFrame({
        "EntryDate": [pd.Timestamp("2020-01-01")],
        "ExitDate": [pd.Timestamp("2022-01-01")]
    })
    out = compute_holding_periods(df)
    # should be ~2 years
    assert abs(out.loc[0, "HoldingYears"] - 2.0) < 0.01
