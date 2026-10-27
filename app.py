import pandas as pd
import numpy as np

# Define our structural assets
sectors = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

print("⏳ Stage 1: Initiating Data Ingestion...")

# 1. Load raw macro data from your folder
df_growth = pd.read_csv("INDPRO.csv")
df_infl = pd.read_csv("CPILFESL.csv")

# 2. Standardize dates and sort oldest-to-newest
df_growth['Date'] = pd.to_datetime(df_growth['observation_date'])
df_growth = df_growth.sort_values('Date').reset_index(drop=True)

df_infl['Date'] = pd.to_datetime(df_infl['observation_date'])
df_infl = df_infl.sort_values('Date').reset_index(drop=True)

# 3. Merge them into a single core macro data frame
macro_df = pd.merge(df_growth[['Date', 'INDPRO']], df_infl[['Date', 'CPILFESL']], on='Date', how='outer')
macro_df = macro_df.sort_values('Date').reset_index(drop=True)

print("✅ Macro files aligned successfully.")
print(macro_df.head(3))

print("\n⏳ Stage 2: Processing Sector ETF Data & Monthly Returns...")

sector_dfs = []

for ticker in sectors:
    # Load the specific sector file
    df_sector = pd.read_csv(f"{ticker}.csv")

    # Standardize dates and sort chronologically
    df_sector['Date'] = pd.to_datetime(df_sector['Date'])
    df_sector = df_sector.sort_values('Date').reset_index(drop=True)

    # Calculate Monthly Return: (Current Adj Close / Previous Adj Close) - 1
    df_sector[f'{ticker}_Return'] = df_sector['Adj Close'].pct_change()

    # Keep only the columns we need to prevent clutter
    sector_dfs.append(df_sector[['Date', f'{ticker}_Return']])

# Merge all 11 sector return columns into a unified data frame
merged_sectors = sector_dfs[0]
for next_df in sector_dfs[1:]:
    merged_sectors = pd.merge(merged_sectors, next_df, on='Date', how='outer')

print("✅ All 11 sector ETF monthly returns calculated and unified.")

print("\n🔍 --- STAGE 1 VERIFICATION REPORT ---")
print(f"Macro Data Shape (Rows, Columns): {macro_df.shape}")
print(f"Macro Columns: {macro_df.columns.tolist()}")
print("----------------------------------------")
print(f"Sector Returns Shape (Rows, Columns): {merged_sectors.shape}")
print(f"Sector Columns: {merged_sectors.columns.tolist()}")
print("=======================================")

print("\n⏳ Stage 2: Engineering Macro Momentum Scores")

# 1. GROWTH MOMENTUM VECTOR (INDPRO)
macro_df['Growth_3M_MA'] = macro_df['INDPRO'].rolling(window=3).mean()
macro_df['Growth_12M_MA'] = macro_df['INDPRO'].rolling(window=12).mean()
# Score > 0 means growth is accelerating; Score <= 0 means decelerating
macro_df['Growth_Score'] = macro_df['Growth_3M_MA'] - macro_df['Growth_12M_MA']

# 2. INFLATION MOMENTUM VECTOR (CPILFESL)
# Step A: Transform raw index into a YoY percentage rate
macro_df['YoY_Inflation'] = macro_df['CPILFESL'].pct_change(12)
# Step B: Calculate moving averages on the YoY rate
macro_df['Inflation_3M_MA'] = macro_df['YoY_Inflation'].rolling(window=3).mean()
macro_df['Inflation_12M_MA'] = macro_df['YoY_Inflation'].rolling(window=12).mean()
# Score > 0 means inflation is accelerating; Score <= 0 means decelerating
macro_df['Inflation_Score'] = macro_df['Inflation_3M_MA'] - macro_df['Inflation_12M_MA']

print("✅ Mathematical momentum scores calculated.")

# 3. DEFINE THE REGIME ASSIGNMENT LOGIC
def assign_quadrant(row):
    # If we don't have enough data history yet for the 12M moving averages, return NaN
    if pd.isna(row['Growth_Score']) or pd.isna(row['Inflation_Score']):
        return np.nan

    g = row['Growth_Score']
    i = row['Inflation_Score']

    if g > 0 and i > 0:
        return "High GDP / High Inflation"
    elif g <= 0 and i > 0:
        return "Low GDP / High Inflation"
    elif g <= 0 and i <= 0:
        return "Low GDP / Low Inflation"
    else:  # g > 0 and i <= 0
        return "High GDP / Low Inflation"

# Apply the logic engine across the dataframe
macro_df['Quadrant'] = macro_df.apply(assign_quadrant, axis=1)

print("✅ 4-Quadrant macro sorting complete.")

print("\n🔍 --- STAGE 2 VERIFICATION REPORT ---")
print("Distribution of Historical Macro Quadrants:")
print(macro_df['Quadrant'].value_counts(dropna=False))
print("=======================================")

print("\n⏳ Stage 3: Running Quantitative Sector Backtest...")

# 1. MERGE MACRO REGIMES WITH ETF RETURNS
# We use an 'inner' merge to automatically drop any dates that don't overlap
master_df = pd.merge(macro_df, merged_sectors, on='Date', how='inner')

# Drop the 23 structural NaN rows from our evaluation pool
master_df = master_df.dropna(subset=['Quadrant']).reset_index(drop=True)

# 2. INITIALIZE ARRAYS TO HOLD RESULTS
quadrants = [
    "High GDP / High Inflation",
    "Low GDP / High Inflation",
    "Low GDP / Low Inflation",
    "High GDP / Low Inflation"
]

avg_returns = {}
hit_rates = {}

# 3. LOOP THROUGH REGIMES AND CALCULATE PERFORMANCE
for q in quadrants:
    # Filter our database for just the months that match the current quadrant
    q_df = master_df[master_df['Quadrant'] == q]

    avg_returns[q] = {}
    hit_rates[q] = {}

    for ticker in sectors:
        ret_col = f'{ticker}_Return'

        # Pull valid return streams for the sector in this quadrant
        valid_rets = q_df[ret_col].dropna()

        if len(valid_rets) > 0:
            # Calculate Mean Monthly Return
            avg_returns[q][ticker] = valid_rets.mean()
            # Calculate Hit Rate (Months with positive returns / total months)
            hit_rates[q][ticker] = (valid_rets > 0).sum() / len(valid_rets)
        else:
            avg_returns[q][ticker] = np.nan
            hit_rates[q][ticker] = np.nan

# 4. CONVERT RAW ARRAYS INTO PRESENTATION MATRICES (%)
df_avg_returns = pd.DataFrame(avg_returns) * 100
df_hit_rates = pd.DataFrame(hit_rates) * 100

print("\n=======================================================")
print("     MATRIX 1: AVERAGE MONTHLY RETURNS (%)")
print("=======================================================")
print(df_avg_returns.round(2))

print("\n=======================================================")
print("     MATRIX 2: REGIME WIN/HIT RATES (%)")
print("=======================================================")
print(df_hit_rates.round(2))

# 5. EXPORT DELIVERABLE TO YOUR DIRECTORY
df_avg_returns.to_excel("Tactical_Sector_Returns.xlsx")
print("\n💾 Success! Your results have been exported to 'Tactical_Sector_Returns.xlsx'.")

# Create a clean, historical timeline ledger of our macro regimes
historical_ledger = master_df[['Date', 'INDPRO', 'Growth_Score', 'CPILFESL', 'Inflation_Score', 'Quadrant']]

# Save it to a separate tab in your Excel file
with pd.ExcelWriter("Tactical_Sector_Returns.xlsx", mode='a', engine='openpyxl') as writer:
    historical_ledger.to_excel(writer, sheet_name="Historical_Timeline", index=False)

print("📅 Historical timeline with exact dates exported to the 'Historical_Timeline' tab!")
