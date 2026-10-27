from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DATA_DIR = Path(".")  # Place the supplied CSVs beside this notebook, or change this directory.
OUTPUT = Path("Quarterly_Quadrant_Results.xlsx")
SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
REGIMES = ["Reflation", "Goldilocks", "Stagflation", "Disinflation / Slowdown"]
COLORS = {"Reflation": "#2e9c64", "Goldilocks": "#377ed0", "Stagflation": "#d95757", "Disinflation / Slowdown": "#d4a536"}

def locate(name):
    candidates = [DATA_DIR / f"{name}.csv"] + sorted(DATA_DIR.glob(f"{name}(*)*.csv"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {name}.csv in {DATA_DIR.resolve()}")

def read_series(name):
    df = pd.read_csv(locate(name))
    date_col = "observation_date" if "observation_date" in df.columns else "Date"
    df["Date"] = pd.to_datetime(df[date_col])
    df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    return df.set_index("Date")[[name]]


monthly = read_series("INDPRO").join(read_series("CPILFESL"), how="inner").sort_index()
monthly = monthly.asfreq("MS")  # Missing months remain missing; never silently interpolate.
monthly["Growth_3M_MA"] = monthly["INDPRO"].rolling(3, min_periods=3).mean()
monthly["Core_CPI_MoM_Pct"] = monthly["CPILFESL"].pct_change(fill_method=None) * 100
monthly["Inflation_3M_MA"] = monthly["Core_CPI_MoM_Pct"].rolling(3, min_periods=3).mean()

# Require the actual last calendar month of the quarter, not the latest available earlier month.
q_months = monthly.loc[monthly.index.month.isin([3, 6, 9, 12])].copy()
q_months.index = q_months.index.to_period("Q")
quarterly = q_months[["INDPRO", "CPILFESL", "Growth_3M_MA", "Inflation_3M_MA"]].copy()
quarterly.index.name = "Quarter"
quarterly["Growth_Score"] = quarterly["Growth_3M_MA"].diff()
quarterly["Inflation_Score"] = quarterly["Inflation_3M_MA"].diff()

def regime(row):
    if pd.isna(row["Growth_Score"]) or pd.isna(row["Inflation_Score"]):
        return np.nan
    g, i = row["Growth_Score"], row["Inflation_Score"]
    if g > 0 and i > 0: return "Reflation"
    if g > 0 and i <= 0: return "Goldilocks"
    if g <= 0 and i > 0: return "Stagflation"
    return "Disinflation / Slowdown"

quarterly["Regime"] = quarterly.apply(regime, axis=1)
quarterly["Quarter_End"] = quarterly.index.to_timestamp(how="end").normalize()
print("Quarterly macro regimes:", quarterly["Regime"].value_counts(dropna=False).to_dict())
display(quarterly.tail(8))

sector_returns = pd.DataFrame()
for ticker in SECTORS:
    df = pd.read_csv(locate(ticker))
    df["Date"] = pd.to_datetime(df["Date"])
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
    # Monthly files use first-of-month date labels; retain only the calendar quarter's final month.
    endpoint = df.loc[df.index.month.isin([3, 6, 9, 12]), "Adj Close"].copy()
    endpoint.index = endpoint.index.to_period("Q")
    endpoint = endpoint.reindex(quarterly.index)  # No filling across missing quarters.
    sector_returns[ticker] = endpoint.pct_change(fill_method=None) * 100

master = quarterly.join(sector_returns)
master = master.dropna(subset=["Regime"])
print("Quarterly classified observations:", len(master))
print("Sector observations by ticker:")
print(master[SECTORS].count())

mean_returns = master.groupby("Regime")[SECTORS].mean().reindex(REGIMES).T
positive_hit_rate = master.groupby("Regime")[SECTORS].agg(lambda x: (x.dropna() > 0).mean() * 100 if x.notna().any() else np.nan).reindex(REGIMES).T
observation_counts = master.groupby("Regime")[SECTORS].count().reindex(REGIMES).T
sector_benchmark = master[SECTORS].mean(axis=1, skipna=True)
relative_win = master[SECTORS].gt(sector_benchmark, axis=0).where(master[SECTORS].notna())
relative_win["Regime"] = master["Regime"]
outperformance_rate = relative_win.groupby("Regime")[SECTORS].mean().reindex(REGIMES).T * 100





forward = master[["Regime"]].join(sector_returns[SECTORS].shift(-1).add_prefix("NextQ_"))
forward_mean = forward.groupby("Regime")[[f"NextQ_{s}" for s in SECTORS]].mean().reindex(REGIMES).T
forward_mean.index = SECTORS

# Transition matrix: only consecutive calendar quarters count.
regime_pairs = pd.DataFrame({"Current": master["Regime"], "Next": master["Regime"].shift(-1), "Next_Quarter": pd.Series(master.index.shift(1), index=master.index)})
regime_pairs = regime_pairs.loc[regime_pairs["Next_Quarter"] == regime_pairs.index + 1]
transitions = pd.crosstab(regime_pairs["Current"], regime_pairs["Next"]).reindex(index=REGIMES, columns=REGIMES, fill_value=0)
transition_probs = transitions.div(transitions.sum(axis=1).replace(0, np.nan), axis=0) * 100

with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    quarterly.reset_index().to_excel(writer, sheet_name="Quarterly_Regimes", index=False)
    master.reset_index().to_excel(writer, sheet_name="Master_Quarterly", index=False)
    mean_returns.to_excel(writer, sheet_name="Avg_Quarterly_Returns")
    positive_hit_rate.to_excel(writer, sheet_name="Positive_Hit_Rates")
    outperformance_rate.to_excel(writer, sheet_name="Relative_Win_Rates")
    observation_counts.to_excel(writer, sheet_name="Observation_Counts")
    transition_probs.to_excel(writer, sheet_name="Regime_Transitions")
    forward_mean.to_excel(writer, sheet_name="Next_Q_Exploratory")
print("Exported:", OUTPUT.resolve())
print("Core calculations complete. Run any visualization cell below.")
