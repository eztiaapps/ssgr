import pandas as pd
import datetime
from io import BytesIO
from collections import defaultdict

def read_process_excel(uploaded_file):
    # Convert Streamlit uploaded file into a BytesIO object
    file_stream = BytesIO(uploaded_file.getvalue())

    # Read Excel file WITHOUT auto-parsing and ensuring first column is not lost
    df = pd.read_excel(file_stream, header=None, skiprows=14, sheet_name="Data Sheet", dtype=object, keep_default_na=False)

    # Convert all empty strings and 'NaT' values to NaN
    df.replace(['', 'NaT'], pd.NA, inplace=True)

    # Drop rows where all values are NaN, NaT, or None
    df.dropna(how='all', inplace=True)

    # Ensure first column is not lost (reset index)
    df.reset_index(drop=True, inplace=True)

    # Convert only "Report Date" rows to date format
    df.loc[df[0] == 'Report Date', 1:] = df.loc[df[0] == 'Report Date', 1:].apply(pd.to_datetime, errors='coerce').apply(lambda x: x.dt.strftime('%Y-%m-%d'))

    # Replace <NA> with 0 when the row contains other valid values
    for idx, row in df.iterrows():
        if row.notna().any():  # Check if at least one value is not NaN
            df.loc[idx] = row.fillna(0)

    # Ensure all other columns retain original values (convert any misparsed datetime back to string)
    for col in df.columns[1:]:  # Skip the first column (categories)
        df[col] = df[col].astype(str)

    # Add unique numbering to duplicate row names in the first column
    name_count = defaultdict(int)
    for idx, value in enumerate(df[0]):
        if pd.notna(value):  # Ensure value is not NaN
            name_count[value] += 1
            if name_count[value] > 1:  # Only modify duplicates
                df.at[idx, 0] = f"{value} {name_count[value]}"

    # **Set the first column as the index**
    df.set_index(0, inplace=True)

    # **Calculate YearlyExpenses**
    required_rows = [
        "Raw Material Cost", "Power and Fuel", "Other Mfr. Exp", 
        "Employee Cost", "Selling and admin", "Other Expenses", "Change in Inventory"
    ]

    # Ensure missing rows are treated as 0
    for row in required_rows:
        if row not in df.index:
            df.loc[row] = 0  # Fill missing rows with 0

    # 1 Perform the YearlyExpense formula calculation
    df.loc["YearlyExpenses"] = (
        df.loc["Raw Material Cost"].astype(float) + 
        df.loc["Power and Fuel"].astype(float) + 
        df.loc["Other Mfr. Exp"].astype(float) + 
        df.loc["Employee Cost"].astype(float) + 
        df.loc["Selling and admin"].astype(float) + 
        df.loc["Other Expenses"].astype(float) - 
        df.loc["Change in Inventory"].astype(float)
    ).astype(str)  # Convert back to string for consistency


    # 2 Perform the Operating Profit calculation
    if "Sales" in df.index and "YearlyExpenses" in df.index:
        df.loc["Operating Profit"] = df.loc["Sales"].astype(float) - df.loc["YearlyExpenses"].astype(float)

    
    # 3 Calculate Operating Profit Margin**
    if "Sales" in df.index and "Operating Profit" in df.index:
        df.loc["OPM%"] = df.loc["Operating Profit"].astype(float) / df.loc["Sales"].astype(float)

    # 4 Calculate NPM% Calculation
    if "Sales" in df.index and "Net profit" in df.index:
        df.loc["NPM%"] = df.loc["Net profit"].astype(float) / df.loc["Sales"].astype(float)

    # 5 Calculate 3 Years Avg NPM% 
    # 5 Calculate Average 3 Year NPM% for each column
    if "NPM%" in df.index:
        npms = df.loc[["NPM%"]].astype(float).fillna(0)  # Convert to DataFrame & replace NaN with 0

        # Compute rolling sum and count valid entries for division
        rolling_sum = npms.rolling(window=3, min_periods=1, axis=1).sum()

        avg_npm = rolling_sum / 3  # Proper division

        # Assign back to DataFrame ensuring shape matches
        df.loc["Average 3 Year NPM%"] = avg_npm.iloc[0]  # Extract the row correctly
    
    
    # 6 Calculate NFAT (Net Fixed Asset Turnover)
    if "Sales" in df.index and "Net Block" in df.index:
        net_block = df.loc[["Net Block"]].astype(float).fillna(0)  # Convert to DataFrame & handle NaNs

        # Compute rolling average of previous period
        avg_net_block = net_block.rolling(window=2, min_periods=1, axis=1).mean()

        # Compute NFAT as Sales / Average Net Block
        df.loc["NFAT"] = df.loc["Sales"].astype(float) / avg_net_block.iloc[0]  # Extract row properly
    
    # 7 Calculate "Average NFAT 3 Years"
    if "NFAT" in df.index:
        nfat = df.loc[["NFAT"]].astype(float).fillna(0)  # Ensure numeric type & handle NaNs

        # Compute rolling average over the last 3 years (including current year)
        avg_nfat_3y = nfat.rolling(window=3, min_periods=1, axis=1).mean()

        # Store result in the dataframe
        df.loc["Average NFAT 3 Years"] = avg_nfat_3y.iloc[0]  # Extract row properly
    
    # 8 Calculate "Dep%"
    if "Depreciation" in df.index and "Net Block" in df.index:
        # Convert values to float and handle NaNs (if any)
        depreciation = df.loc["Depreciation"].astype(float).fillna(0)
        net_block = df.loc["Net Block"].astype(float).fillna(0)

        # Calculate Dep% (Depreciation / Net Block)
        df.loc["Dep%"] = depreciation / net_block
    
    # 8 Calculate "Dep%"
    if "Dep%" in df.index:
        deps = df.loc[["Dep%"]].astype(float).fillna(0)  # Convert to DataFrame & replace NaN with 0

        # Compute rolling sum and count valid entries for division
        rolling_sum_dep = deps.rolling(window=3, min_periods=1, axis=1).sum()

        avg_dep = rolling_sum_dep / 3  # Proper division

        # Assign back to DataFrame ensuring shape matches
        df.loc["Average 3 Year Dep%"] = avg_dep.iloc[0]  # Extract the row correctly
    
    # 9 Calculate "DPR" (Dividend Payout Ratio)
    if "Dividend Amount" in df.index and "Net profit" in df.index:
        # Convert to numeric, replace NaNs with 0
        dividend_amount = df.loc["Dividend Amount"].astype(float).fillna(0)
        net_profit = df.loc["Net profit"].astype(float).fillna(0)

        # Avoid division by 0 by replacing 0/0 with NaN or 0
        dpr = dividend_amount / net_profit
        dpr = dpr.replace([float('inf'), -float('inf')], 0)  # Replace any infinite values with 0

        # Store result in the dataframe
        df.loc["DPR"] = dpr

    # 10 Calculate "Dep%"
    if "DPR" in df.index:
        dprs = df.loc[["DPR"]].astype(float).fillna(0)  # Convert to DataFrame & replace NaN with 0

        # Compute rolling sum and count valid entries for division
        rolling_sum_dpr = dprs.rolling(window=3, min_periods=1, axis=1).sum()

        avg_dpr = rolling_sum_dpr / 3  # Proper division

        # Assign back to DataFrame ensuring shape matches
        df.loc["Average 3 Year DPR"] = avg_dpr.iloc[0]  # Extract the row correctly

    # 11 Calculate "BSR" (Business Success Ratio)
    if "Average NFAT 3 Years" in df.index and "Average 3 Year NPM%" in df.index and "Average 3 Year DPR" in df.index and "Average 3 Year Dep%" in df.index:
        # Extract necessary rows as float
        avg_nfat_3y = df.loc["Average NFAT 3 Years"].astype(float).fillna(0)
        avg_npm_3y = df.loc["Average 3 Year NPM%"].astype(float).fillna(0)
        avg_dpr_3y = df.loc["Average 3 Year DPR"].astype(float).fillna(0)
        avg_dep_3y = df.loc["Average 3 Year Dep%"].astype(float).fillna(0)

        # Calculate BSR (Business Success Ratio)
        bsr = avg_nfat_3y * avg_npm_3y * (1 - avg_dpr_3y) - avg_dep_3y

        # Store result in the dataframe
        df.loc["BSR"] = bsr

    
    return df





def read_process_excel_bkp(uploaded_file):
    # Convert Streamlit uploaded file into a BytesIO object
    file_stream = BytesIO(uploaded_file.getvalue())

    # Read Excel file WITHOUT auto-parsing and ensuring first column is not lost
    df = pd.read_excel(file_stream, header=None, skiprows=14, sheet_name="Data Sheet", dtype=object, keep_default_na=False)

    # Convert all empty strings and 'NaT' values to NaN
    df.replace(['', 'NaT'], pd.NA, inplace=True)

    # Drop rows where all values are NaN, NaT, or None
    df.dropna(how='all', inplace=True)

    # Ensure first column is not lost (reset index)
    df.reset_index(drop=True, inplace=True)

    # Convert only "Report Date" rows to date format
    df.loc[df[0] == 'Report Date', 1:] = df.loc[df[0] == 'Report Date', 1:].apply(pd.to_datetime, errors='coerce').apply(lambda x: x.dt.strftime('%Y-%m-%d'))

    # Replace <NA> with 0 when the row contains other valid values
    for idx, row in df.iterrows():
        if row.notna().any():  # Check if at least one value is not NaN
            df.loc[idx] = row.fillna(0)

    # Ensure all other columns retain original values (convert any misparsed datetime back to string)
    for col in df.columns[1:]:  # Skip the first column (categories)
        df[col] = df[col].astype(str)

    # Add unique numbering to duplicate row names in the first column
    name_count = defaultdict(int)
    for idx, value in enumerate(df[0]):
        if pd.notna(value):  # Ensure value is not NaN
            name_count[value] += 1
            if name_count[value] > 1:  # Only modify duplicates
                df.at[idx, 0] = f"{value} {name_count[value]}"

    # **Set the first column as the index**
    df.set_index(0, inplace=True)

    # **Calculate YearlyExpenses**
    required_rows = [
        "Raw Material Cost", "Power and Fuel", "Other Mfr. Exp", 
        "Employee Cost", "Selling and admin", "Other Expenses", "Change in Inventory"
    ]

    # Ensure missing rows are treated as 0
    for row in required_rows:
        if row not in df.index:
            df.loc[row] = 0  # Fill missing rows with 0

    # 1 Perform the YearlyExpense formula calculation
    df.loc["YearlyExpenses"] = (
        df.loc["Raw Material Cost"].astype(float) + 
        df.loc["Power and Fuel"].astype(float) + 
        df.loc["Other Mfr. Exp"].astype(float) + 
        df.loc["Employee Cost"].astype(float) + 
        df.loc["Selling and admin"].astype(float) + 
        df.loc["Other Expenses"].astype(float) - 
        df.loc["Change in Inventory"].astype(float)
    ).astype(str)  # Convert back to string for consistency


    # 2 Perform the Operating Profit calculation
    if "Sales" in df.index and "YearlyExpenses" in df.index:
        df.loc["Operating Profit"] = df.loc["Sales"].astype(float) - df.loc["YearlyExpenses"].astype(float)

    
    # 3 Calculate Operating Profit Margin**
    if "Sales" in df.index and "Operating Profit" in df.index:
        df.loc["OPM%"] = df.loc["Operating Profit"].astype(float) / df.loc["Sales"].astype(float)

    # 4 Calculate NPM% Calculation
    if "Sales" in df.index and "Net profit" in df.index:
        df.loc["NPM%"] = df.loc["Net profit"].astype(float) / df.loc["Sales"].astype(float)

    # 5 Calculate 3 Years Avg NPM% 
    # 5 Calculate Average 3 Year NPM% for each column
    if "NPM%" in df.index:
        npms = df.loc[["NPM%"]].astype(float).fillna(0)  # Convert to DataFrame & replace NaN with 0

        # Compute rolling sum and count valid entries for division
        rolling_sum = npms.rolling(window=3, min_periods=1, axis=1).sum()

        avg_npm = rolling_sum / 3  # Proper division

        # Assign back to DataFrame ensuring shape matches
        df.loc["Average 3 Year NPM%"] = avg_npm.iloc[0]  # Extract the row correctly
    
    
    # 6 Calculate NFAT (Net Fixed Asset Turnover)
    if "Sales" in df.index and "Net Block" in df.index:
        net_block = df.loc[["Net Block"]].astype(float).fillna(0)  # Convert to DataFrame & handle NaNs

        # Compute rolling average of previous period
        avg_net_block = net_block.rolling(window=2, min_periods=1, axis=1).mean()

        # Compute NFAT as Sales / Average Net Block
        df.loc["NFAT"] = df.loc["Sales"].astype(float) / avg_net_block.iloc[0]  # Extract row properly
    
    # 7 Calculate "Average NFAT 3 Years"
    if "NFAT" in df.index:
        nfat = df.loc[["NFAT"]].astype(float).fillna(0)  # Ensure numeric type & handle NaNs

        # Compute rolling average over the last 3 years (including current year)
        avg_nfat_3y = nfat.rolling(window=3, min_periods=1, axis=1).mean()

        # Store result in the dataframe
        df.loc["Average NFAT 3 Years"] = avg_nfat_3y.iloc[0]  # Extract row properly



    

    return df





