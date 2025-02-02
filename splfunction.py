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

    # Perform the formula calculation
    df.loc["YearlyExpenses"] = (
        df.loc["Raw Material Cost"].astype(float) + 
        df.loc["Power and Fuel"].astype(float) + 
        df.loc["Other Mfr. Exp"].astype(float) + 
        df.loc["Employee Cost"].astype(float) + 
        df.loc["Selling and admin"].astype(float) + 
        df.loc["Other Expenses"].astype(float) - 
        df.loc["Change in Inventory"].astype(float)
    ).astype(str)  # Convert back to string for consistency

    return df



#Again higher the version, it is working. Don't delete it.
def read_process_excel3(uploaded_file):
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

    return df



#Don't delete this function. It is working. TODO
def read_process_excel2(uploaded_file):
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

    return df



def read_process_excel_backup(uploaded_file):
    # Convert Streamlit uploaded file into a BytesIO object
    file_stream = BytesIO(uploaded_file.getvalue())

    # Read Excel file WITHOUT auto-parsing and ensuring first column is not lost
    df = pd.read_excel(file_stream, header=None, skiprows=14, sheet_name="Data Sheet", dtype=object, keep_default_na=False)

    # Drop rows where all values are NaN, NaT, or None
    # Convert all empty strings and 'NaT' values to NaN
    df.replace(['', 'NaT'], pd.NA, inplace=True)

    # Drop rows where ALL values are NaN, NaT, or None
    df.dropna(how='all', inplace=True)

    # Ensure first column is not lost
    df.reset_index(drop=True, inplace=True)  # Reset index to prevent shifting issues

    # Iterate over the rows to convert <NA> to 0 when there are other valid values
    for idx, row in df.iterrows():
        if row.notna().any():  # Check if any value in the row is not NaN or NaT
            # Replace <NA> with 0 for the entire row if it's not all NaN/NaT
            df.loc[idx] = row.fillna(0)
    
    # Ensure all other columns retain original values (convert any misparsed datetime back to string)
    for col in df.columns[1:]:  # Skip Report Date column
        df[col] = df[col].astype(str)

    
    return df

