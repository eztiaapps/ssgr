import pandas as pd
import datetime
from io import BytesIO
from collections import defaultdict
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from metrics import POSITIVE_METRICS, NEGATIVE_METRICS
import os

#Read the csv balance sheet, and process the file and parse it. Then calculates additionals rows of information like BSR.
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
        df.loc["BSR"] = bsr * 100

    
    return df




def calculate_growth(data):
    """Calculate year-over-year growth for a given data series"""
    growth = data.pct_change() * 100  # percentage change from the previous year
    growth = growth.fillna(0)  # Fill NaN values (for the first year) with 0
    return growth

def plot_growth_vs_bsr(df):
    # Extract the "Report Date" row to use as the X-axis labels (years or specific dates)
    report_dates = pd.to_datetime(df.loc["Report Date"].dropna(), errors='coerce')  # Drop NaT values

    # Extract the corresponding trend data for Sales, Yearly Expenses, and BSR
    sales = df.loc["Sales"].astype(float)
    yearly_expenses = df.loc["YearlyExpenses"].astype(float)
    bsr = df.loc["BSR"].astype(float)

    # Calculate the growths
    sales_growth = calculate_growth(sales)
    yearly_expenses_growth = calculate_growth(yearly_expenses)
    bsr_growth = calculate_growth(bsr)


    # Ensure the lengths match
    if len(report_dates) != len(sales_growth) or len(report_dates) != len(yearly_expenses_growth) or len(report_dates) != len(bsr):
        raise ValueError("The length of Report Date and trend data do not match.")
    
    # Sort the data by date to ensure proper alignment
    sorted_dates = sorted(zip(report_dates, sales_growth, yearly_expenses_growth, bsr))
    report_dates, sales_growth, yearly_expenses_growth, bsr = zip(*sorted_dates)

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot the growth trends as lines
    plt.plot(report_dates, sales_growth, label="Sales Growth", color='#00bfae', linewidth=3)  # Bright cyan-like color
    plt.plot(report_dates, yearly_expenses_growth, label="Yearly Expenses Growth", color='#ff4081', linewidth=3)  # Bright pink
    plt.plot(report_dates, bsr, label="BSR", color='#ffeb3b', linewidth=3)  # Bright yellow

    # Customize font properties for a modern look
    plt.title("Growth vs BSR: Sales & Yearly Expenses Growth", fontsize=18, color='white', fontweight='bold')
    plt.xlabel("Year", fontsize=14, color='white')
    plt.ylabel("Growth / BSR", fontsize=14, color='white')

    # Set legend with white labels and transparent background
    plt.legend(loc='upper left', fontsize=12, frameon=False, labels=['Sales Growth', 'Yearly Expenses Growth', 'BSR'], labelcolor='white')

    # Set grid lines with white and more contrast
    plt.grid(True, which='both', axis='y', linestyle='-', color='white', alpha=0.3)

    # Set figure and axes backgrounds to transparent
    plt.gca().patch.set_facecolor('none')  # Transparent axes background
    plt.gcf().patch.set_facecolor('none')  # Transparent figure background

    # Apply white color to ticks for a clean look
    plt.tick_params(axis='both', labelsize=12, labelcolor='white', colors='white')

    # Show the plot in Streamlit
    st.pyplot(plt)


def ReportSummary(df):
    st.write("TODO: Report Summary")


def calculate_growth_score(df, metric, years=5):
    """Calculate a weighted growth score for a given metric over a specified period."""

    if metric not in df.index or "Report Date" not in df.index:
        return None  # Return None if data or metric is missing

    # Convert 'Report Date' to datetime and sort by date
    report_dates = pd.to_datetime(df.loc["Report Date"], errors='coerce')
    metric_data = df.loc[metric].astype(float)

    # Drop columns with invalid dates
    valid_mask = report_dates.notna()
    report_dates = report_dates[valid_mask]
    metric_data = metric_data[valid_mask]

    # Sort data in chronological order (oldest to latest)
    sorted_indices = report_dates.argsort()
    report_dates = report_dates.iloc[sorted_indices]
    metric_data = metric_data.iloc[sorted_indices]

    # Determine the latest available year
    latest_year = report_dates.dt.year.max()
    target_years = list(range(latest_year, latest_year - years, -1))

    # Filter data to only include the target years
    metric_filtered = metric_data[report_dates.dt.year.isin(target_years)]
    report_filtered = report_dates[report_dates.dt.year.isin(target_years)]

    # Ensure we have enough periods for calculation
    periods = min(len(metric_filtered) - 1, years - 1)
    if periods < 2:
        return None  # Not enough data

    # Calculate Year-over-Year Growth (%)
    growth_rates = []
    for i in range(1, periods + 1):
        prev_value = metric_filtered.iloc[i - 1]
        curr_value = metric_filtered.iloc[i]

        if prev_value == 0:
            growth_rate = 0  # Avoid division by zero
        else:
            growth_rate = ((curr_value - prev_value) / abs(prev_value)) * 100

        # If it's a negative metric, invert the impact (growth is bad, reduction is good)
        if metric in NEGATIVE_METRICS:
            growth_rate = -growth_rate  # Invert logic: Higher growth is bad, reduction is good

        growth_rates.append(growth_rate)

    # Apply weighting (more recent years have higher weight)
    weights = np.arange(periods, 0, -1)  # Example: [5, 4, 3, 2, 1] for 5 years
    weighted_growth = np.dot(growth_rates, weights) / weights.sum()

    # Normalize score between 0-100 (assuming min/max range)
    min_score, max_score = -50, 50  # Define reasonable bounds for scaling
    normalized_score = np.clip((weighted_growth - min_score) / (max_score - min_score) * 100, 0, 100)

    return round(normalized_score, 2)


def display_score(metric, df, years):
    """Display growth score with color-coded rating."""
    score = calculate_growth_score(df, metric, years)
    
    if score is not None:
        # Adjust for negative impact metrics
        if metric in NEGATIVE_METRICS:
            score = 100 - score  # Invert score for metrics where lower is better

        score_value = st.slider(
            f"{metric} Growth Score ({years} Years)", 0, 100, int(score),
            disabled=True, format="%d"
        )
        
        # Assign color based on score
        if score_value < 50:
            color = "red"
            label = "Poor"
        elif score_value < 75:
            color = "orange"
            label = "Average"
        else:
            color = "green"
            label = "Good"
        
        # Display colored rating
        st.markdown(f"<span style='color:{color}; font-size:18px; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
        return score
    else:
        st.warning(f"Not enough data for {metric} ({years} Years)")
        return None

def calculate_overall_score(scores):
    """Compute the overall score as an average of available metric scores."""
    valid_scores = [s for s in scores if s is not None]
    
    if valid_scores:
        return round(sum(valid_scores) / len(valid_scores), 2)
    return None

def colored_text(text, color):
    return f"<p style='color:{color}; font-size:18px;'>{text}</p>"


# Function to read the visitor count from file 
counter_file = './visitor_counter.txt'
def get_visitor_count():
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as file:
            return int(file.read())  # Read the count from the file
    else:
        return 0  # If the file doesn't exist, it's the first visit

# Function to increment and update the visitor count
def increment_visitor_count():
    current_count = get_visitor_count()
    new_count = current_count + 1  # Increment the count
    with open(counter_file, 'w') as file:
        file.write(str(new_count))  # Write the updated count back to the file
    return new_count

def contactus():
    """Displays the Contact Us section at the bottom."""
    with st.container():
        st.markdown(
            "<h2 style='text-align: center; font-family: Times New Roman, serif; color: yellow;'>📞 Contact Us</h2>",
            unsafe_allow_html=True
        )

        st.write("📧 **Email:** support@datalotus.com")
        st.write("📞 **Phone:** +1 234 567 8900")
        st.write("📍 **Address:** 123 Finance Street, Market City, USA")

        # Contact Form
        with st.form(key="contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            message = st.text_area("Your Message")
            submit_button = st.form_submit_button("Submit")

            if submit_button:
                st.success("✅ Thank you! We'll get back to you soon.")




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


COMMENTS_FILE = "comments.txt"

def save_comment(email, phone, comment):
    """Save user feedback to a text file in comma-separated format."""
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = f"{date}, {email}, {phone}, {comment}\n"

    # Append comment to the file
    with open(COMMENTS_FILE, "a") as file:
        file.write(data)