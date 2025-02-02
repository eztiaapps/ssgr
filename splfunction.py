import pandas as pd
import datetime

def split_excel_into_dataframes(file):
    """
    Splits an Excel sheet into multiple DataFrames based on the headers identifying separate sections.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        dict: A dictionary where keys are section names and values are corresponding DataFrames.
    """
    # Load the Excel file
    sheet = pd.read_excel(file, sheet_name='Data Sheet', header=None)

    # Define keywords that identify the start of each data frame
    section_keywords = ["PROFIT & LOSS", "Quarters", "BALANCE SHEET", "CASH FLOW", "DERIVED"]

    dataframes = {}
    current_section = None
    current_data = []

    for index, row in sheet.iterrows():
        # Convert row values to strings for comparison
        row_str = row.astype(str)

        # Check if the row contains a section keyword
        for keyword in section_keywords:
            if row_str.str.contains(keyword, na=False).any():
                # If there's an active section, save the data collected so far
                if current_section and current_section != "PRICE":
                    dataframes[current_section] = pd.DataFrame(current_data)

                # Start a new section
                current_section = keyword
                current_data = []
                break
        else:
            # Handle the PRICE section specifically
                        # Handle the PRICE: section specifically
            if current_section == "PRICE:":
                print('Yes')
                first_cell = str(row[0]).strip() if not pd.isna(row[0]) else ""
                print(first_cell)
                if first_cell.upper() == "PRICE:":
                    # Collect all values after the "PRICE:" cell as a list
                    prices = row.iloc[1:].dropna().tolist()  # Use iloc to explicitly skip the first cell
                    dataframes[current_section] = pd.DataFrame({"Price Values": prices})
                    current_section = None  # Reset as it's a single-row section
                continue


            # Append the row to the current section data if a section is active
            if current_section is not None:
                current_data.append(row.tolist())
                

    # Save the last section
    if current_section and current_section != "PRICE":
        dataframes[current_section] = pd.DataFrame(current_data)
        

    return dataframes

def pnlpreprocess(df):
    # Set the first row (which contains the report date) as the columns
    df.columns = df.iloc[0]

    # Drop the first row
    df = df.drop(index=0).reset_index(drop=True)

    # Drop rows with NaN or None values
    df = df.dropna()  # Remove any row with NaN or None

    # Drop columns with NaN or None values
    df = df.dropna(axis=1, how='any')  # Remove any column with NaN or None

    # Check if the last column is completely NaN or None
    if df.iloc[:, -1].isna().all():
        df = df.drop(columns=df.columns[-1])

    # Function to check if a value is a valid date and remove time
    def clean_date_column(col):
        try:
            # Try to convert to datetime, remove time, and return date
            return pd.to_datetime(col).date()
        except ValueError:
            # If it's not a date, return the original value (like 'Report Date')
            return col

    # Apply the cleaning function to the column names
    df.columns = [clean_date_column(col) for col in df.columns]
    
    return df

# Example usage
# file_path = "path_to_your_file.xlsx"
# dataframes = split_excel_into_dataframes(file_path)
# for name, df in dataframes.items():
#     print(f"DataFrame for section: {name}")
#     print(df.head())

