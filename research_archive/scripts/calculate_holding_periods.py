import pandas as pd
from typing import Tuple

def calculate_holding_period(
    entry_date: pd.Series,
    exit_date: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates the holding period between a set of entry and exit dates.

    The holding period is calculated in both days and years. If the exit date
    is NaT (Not a Time) or missing, the holding period is marked as NaN (Not a Number),
    indicating the deal is still "live" or not yet exited.

    Args:
        entry_date: A pandas Series containing the datetime objects for the entry date.
        exit_date: A pandas Series containing the datetime objects for the exit date.

    Returns:
        A tuple containing two pandas Series:
        1. holding_days: The holding period in days (NaN if not exited).
        2. holding_years: The holding period in years (NaN if not exited).
    """

    # Ensure the inputs are datetime objects for correct subtraction
    entry_date = pd.to_datetime(entry_date, errors='coerce')
    exit_date = pd.to_datetime(exit_date, errors='coerce')

    # Calculate the time difference (Timedelta)
    time_diff = exit_date - entry_date

    # 1. Calculate holding period in days
    holding_days = time_diff.dt.days

    # 2. Calculate holding period in years (assuming 365.25 days/year for simplicity)
    # The .fillna(0) is temporary just to perform the division, the NaT will result in NaN anyway.
    holding_years = holding_days / 365.25

    # Any calculated value where the exit date was NaT will result in NaN, which is correct
    # for deals that are still live.

    return holding_days, holding_years

# Example usage (will not run when imported, only if run directly)
if __name__ == '__main__':
    data = {
        'Entry_Date': ['2018-01-15', '2019-06-01', '2022-03-20', '2020-09-01'],
        'Exit_Date': ['2023-07-20', '2025-12-31', None, '2020-09-01']
    }
    df_test = pd.DataFrame(data)

    df_test['Entry_Date'] = pd.to_datetime(df_test['Entry_Date'])
    df_test['Exit_Date'] = pd.to_datetime(df_test['Exit_Date'])

    days, years = calculate_holding_period(df_test['Entry_Date'], df_test['Exit_Date'])

    df_test['Holding_Days'] = days.round(0).astype('Int64') # Round and convert to nullable integer
    df_test['Holding_Years'] = years.round(4) # Round to 4 decimal places

    print("--- Test Holding Period Calculation ---")
    # FIX: Using floatfmt for better alignment and rounding for cleaner output
    print(df_test.to_string(
        index=False,
        float_format=lambda x: f'{x:.4f}' if not pd.isna(x) else 'NaN',
        na_rep='NaT'
    ))