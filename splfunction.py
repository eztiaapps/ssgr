import pandas as pd
import datetime
from io import BytesIO
from collections import defaultdict
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from metrics import POSITIVE_METRICS, NEGATIVE_METRICS
import os
import uuid
import sqlite3

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

    print(df.loc[0])

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
    
    #12 Calculate EPS
    if "Net profit" in df.index and "Adjusted Equity Shares in Cr" in df.index:
        net_profit = df.loc["Net profit"].astype(float)
        equity_shares = df.loc["Adjusted Equity Shares in Cr"].astype(float)

        # Avoid division by zero
        df.loc["EPS"] = net_profit / equity_shares.replace(0, float("nan"))
        df.loc["EPS"].fillna(0, inplace=True)  # Replace NaN with 0
    
    #13. EPS Growth
    if "EPS" in df.index:
        eps_values = df.loc["EPS"].astype(float)

        # Shift EPS values to get the previous year EPS
        previous_eps = eps_values.shift(1)

        # Standard EPS Growth Calculation
        eps_growth = (eps_values - previous_eps) / previous_eps

        # Handle special case: Previous EPS is negative, Current EPS is positive
        mask = (previous_eps < 0) & (eps_values > 0)
        eps_growth[mask] = ((eps_values - previous_eps) / abs(previous_eps)) * 100

        # Replace NaN and infinity values with 0 to avoid errors
        eps_growth.replace([float("inf"), float("-inf")], 0, inplace=True)
        eps_growth.fillna(0, inplace=True)

        # Store in DataFrame
        df.loc["EPS Growth"] = eps_growth

    # 14. Price to Earning (PE)
    if "EPS" in df.index and "PRICE:" in df.index:
        eps_values = df.loc["EPS"].astype(float)
        price_values = df.loc["PRICE:"].astype(float)

        # Calculate PE Ratio
        pe_ratio = price_values / eps_values

        # Handle division by zero or invalid values
        pe_ratio.replace([float("inf"), float("-inf")], 0, inplace=True)
        pe_ratio.fillna(0, inplace=True)

        # Store in DataFrame
        df.loc["PE"] = pe_ratio
        
    # 15. PE Growth
    if "PE" in df.index:
        pe_values = df.loc["PE"].astype(float)

        # Shift EPS values to get the previous year EPS
        previous_pe = pe_values.shift(1)

        # Standard EPS Growth Calculation
        pe_growth = (pe_values - previous_pe) / previous_pe

        # Handle special case: Previous EPS is negative, Current EPS is positive
        mask = (previous_pe < 0) & (pe_values > 0)
        pe_growth[mask] = ((pe_values - previous_pe) / abs(previous_pe)) * 100

        # Replace NaN and infinity values with 0 to avoid errors
        pe_growth.replace([float("inf"), float("-inf")], 0, inplace=True)
        pe_growth.fillna(0, inplace=True)

        # Store in DataFrame
        df.loc["PE Growth"] = pe_growth

    #16. Rolling PE Growth and EPS Growth 5 and 3 years.
    # Then we can see whether our predictions working or not.
    # When compared with Price.

    
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



def calculate_overall_growth_score(df):
    """Calculate the overall growth score, adjusting based on BSR growth score."""

    # Calculate individual growth scores
    bsr_growth_score = calculate_growth_score(df, "BSR", years=5)
    overall_growth_score = sum(
        calculate_growth_score(df, metric, years=5) or 0
        for metric in df.index if metric not in ["Report Date", "BSR"]
    ) / (len(df.index) - 2)  # Exclude Report Date & BSR

    # Apply BSR influence on overall score
    if bsr_growth_score is not None:
        if bsr_growth_score < 40:
            overall_growth_score = 40  # Poor
        elif bsr_growth_score <= 70:
            overall_growth_score = 50  # Average

    return round(overall_growth_score, 2)




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


def commit_to_git():
    """Automatically commit and push changes to Git."""
    try:
        os.system("git add comments.txt")
        os.system('git commit -m "Auto-update comments"')
        os.system("git push origin main")
        st.info("Yo u can also email us : eztiaapps@gmail.com 🚀")
    except Exception as e:
        st.error(f"Git push failed: {e}")


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

# Function to generate a unique user ID using UUID
def generate_user_id():
    return str(uuid.uuid4())[:8]  # Generates a unique ID and uses the first 8 characters

# Function to store user comments in a file
def save_comment(email, phone, comment):
    user_id = generate_user_id()
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # If phone is empty, store it as an empty string
    phone = phone if phone else ""
    # If comment is empty, store it as an empty string
    comment = comment if comment else ""
    with open("comments.txt", "a") as file:
        file.write(f"{user_id},{email},{phone},{date},{comment}\n")

# Function to read and display comments
def display_comments():
    try:
        with open("comments.txt", "r") as file:
            comments = file.readlines()

        # Display the most recent comments
        if comments:
            st.subheader("Recent Comments")
            for comment in comments:
                # Split the comment data into respective fields
                fields = comment.strip().split(",")
                # Ensure that there are exactly 5 values in each line
                if len(fields) == 5:
                    user_id, email, phone, date, comment_text = fields
                    # Display the comment details
                    st.write(f"**User** : {user_id} : on {date}")
                    st.write(f"Comment: {comment_text}")
                    st.markdown("---")
        else:
            st.write("No comments yet.")
    except FileNotFoundError:
        st.write("No comments yet.")

# Function to display the overall score (for illustration)
def display_overall_score():
    # Example: Show the overall score (replace with your actual score calculation)
    overall_score = 85  # Placeholder for overall score
    st.subheader("Overall Score")
    st.write(f"**Score: {overall_score}**")
    st.markdown("---")

# Function for the comment section below the overall score
def comment_section():
    email = st.text_input("Email (Required)", key="email")
    phone = st.text_input("Phone (Optional)", key="phone")
    comment = st.text_area("Your Comment", key="comment")

    if st.button("Submit Comment"):
        if email:
            save_comment(email, phone, comment)
            st.success("Your comment has been submitted!")
        else:
            st.error("Email is required to submit a comment.")




# Function to get PE and EPS current from the DataFrame
def get_pe_eps_current(df):
    """Retrieve the most recent PE and EPS values from the DataFrame."""
    try:
        # Get the most recent PE value (from the 'PE' row)
        pe_current = df.loc["PE"].dropna().iloc[-1]  # Get last non-null value in PE row
        
        # Get the most recent EPS value (from the 'EPS' row)
        eps_current = df.loc["EPS"].dropna().iloc[-1]  # Get last non-null value in EPS row
        
        return pe_current, eps_current
    except KeyError:
        # If 'PE' or 'EPS' rows do not exist in the DataFrame
        print("Error: PE or EPS data not found in the DataFrame.")
        return None, None

# Function to calculate Fair Value using PE and EPS CAGR
def get_fair_value_for_available_years(df, pe_current, eps_current, years=5):
    """
    Calculates Fair Value based on available PE and EPS data and CAGR for the available years.
    
    Parameters:
    - df: DataFrame with PE and EPS rows
    - pe_current: Current PE value (from the most recent data)
    - eps_current: Current EPS value (from the most recent data)
    - years: The number of years to calculate the CAGR (default is 5)
    
    Returns:
    - Fair value calculated based on the available data
    """
    # Determine available years for PE and EPS
    available_years = df.shape[1] - 1  # excluding the 'Report Date' column
    if available_years < years:
        print(f"Data available for {available_years} years. Calculating using {available_years} years.")
        years = available_years
    
    # Calculate the PE and EPS CAGR for the available years
    pe_values = df.loc["PE"].dropna()[:years].values
    eps_values = df.loc["EPS"].dropna()[:years].values

    # Calculate the PE and EPS CAGRs
    pe_cagr = calculate_cagr(pe_values)
    eps_cagr = calculate_cagr(eps_values)

    # Calculate the fair value using the formula
    fair_value = (1 + pe_cagr) * pe_current * (1 + eps_cagr) * eps_current
    
    return round(fair_value, 2)


def safe_cagr(value):
    """Converts CAGR to float safely, returns 0 if it's an error message."""
    return value if isinstance(value, (int, float)) else 0

# Function to calculate CAGR for a given set of values
def calculate_cagr_bkp(values):
    """Calculates the Compound Annual Growth Rate (CAGR) given an array of values."""
    if len(values) < 2:
        return 0  # Not enough data
    
    start_value = values[0]
    end_value = values[-1]
    years = len(values) - 1
    
    # If start and end values are the same, return 0 CAGR (no growth)
    if start_value == end_value:
        return 0

    # Handle negative start values
    if start_value < 0 and end_value > 0:
        return "CAGR not meaningful (crosses zero)"

    # Use absolute values for calculation, then restore sign if needed
    abs_cagr = (abs(end_value) / abs(start_value)) ** (1 / years) - 1

    # Restore negative sign if trend remains negative
    if start_value < 0 and end_value < 0:
        return -abs_cagr
    else:
        return abs_cagr


def calculate_cagr(values):
    """Calculates CAGR while handling zero and negative start values safely."""
    if len(values) < 2:
        return 0  # Not enough data
    
    start_value = values[0]
    end_value = values[-1]
    years = len(values) - 1

    # Handle zero or negative starting values
    if start_value == 0:
        return "CAGR not meaningful (start value is zero)"

    if start_value < 0 and end_value > 0:
        return "CAGR not meaningful (crosses zero)"

    try:
        abs_cagr = (abs(end_value) / abs(start_value)) ** (1 / years) - 1
        return -abs_cagr if (start_value < 0 and end_value < 0) else abs_cagr
    except ZeroDivisionError:
        return "CAGR not meaningful (division by zero)"

def get_metric_values_last_n_years(df, metric, n):
    """Retrieve the list of historical values for a given metric (PE, EPS, etc.) for the last n years."""
    try:
        # Ensure the given metric exists in the DataFrame
        if metric not in df.index:
            print(f"Error: '{metric}' row not found in the DataFrame.")
            return []

        # Get the values for the specified metric, drop NaN values
        metric_values = df.loc[metric].dropna()  # Drop any NaN values
        
        # If there are fewer than n values, return all available values
        if len(metric_values) < n:
            print(f"Warning: Only {len(metric_values)} years of data available. Returning all available data.")
            return metric_values.tolist()

        # Return the last n values
        return metric_values[-n:].tolist()
    
    except KeyError:
        # Handle the case if the specified metric row does not exist in the DataFrame
        print(f"Error: '{metric}' row not found in the DataFrame.")
        return []





 # Function to get SQLite connection




#Connect to stocklist db to fetch stock name
def get_connection():
    return sqlite3.connect("stocklist_v1.db", check_same_thread=False)

# Function to fetch stock names from the database
def get_stock_names():
    conn = get_connection()
    query = "SELECT DISTINCT symbol FROM data_table"
    df = pd.read_sql(query, conn)
    conn.close()
    return df["symbol"].tolist()

# Function to fetch stock data based on selection
def get_stock_data(stock_name):
    conn = get_connection()
    query = f"SELECT exchange,symbol,name_of_company,isin_number FROM data_table WHERE symbol = '{stock_name}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df   




