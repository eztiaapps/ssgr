import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from splfunction import read_process_excel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline





#---------------Settings ----------------------#
currency = "INR"
page_title = "DataLotus Dashboard"
page_icon= ":money_with_wings:"
layout = "wide"

#---------------Settings ----------------------#

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)
st.write("""

        # Welcome,
         
         """)

st.write("""
         
         _This app calculates the safety margin, BSR, evaluate key business numbers and helps to avoid bad investments._
         ### ...Beware! If BSR line is below Sales line...!
         """)

"---"


# ----- Sidebar to upload file ----- #
st.sidebar.header('Upload Stock Balance Sheet')
st.sidebar.markdown ("""
                     
                     - ### Step 1: Visit screener.in
                     - ### Step 2: Login to export annual financial report
                     - ### Step 3: Search your stock
                     - ### Step 4: Click the "Export to Excel"
                     """)


# Function to reset state when the file changes (1)
def reset_state():
    st.session_state.df = None  # Reset DataFrame
    st.session_state.chart_drawn = False  # Reset chart status


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

# Track the previously uploaded file (2)
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None


# Collects the stock csv file from user
uploaded_file = st.sidebar.file_uploader("Upload that stock balance sheet here", type=["xlsx"])

# Check if file was removed or changed (3)
if uploaded_file != st.session_state.uploaded_file:
    reset_state()  # Reset everything
    st.session_state.uploaded_file = uploaded_file  # Update file in session state

if uploaded_file is not None:
    df = read_process_excel(uploaded_file)
    file_name =  uploaded_file.name.split('.')[0]

    st.write("1. Checkpoint: Is Sales higher than BSR?")
    st.write(f"""
             # Sales vs Business Sustainability : {file_name}
             """)
    


    #Plot the graph    
    st.write(df)
    "---"

    # Call the plotting function
    plot_growth_vs_bsr(df)

    '''
    "---"

    # Extract the "Report Date" row to use as the X-axis labels (years or specific dates)
    report_dates = pd.to_datetime(df.loc["Report Date"].dropna(), errors='coerce')  # Drop NaT values

    # Extract the corresponding trend data for Sales, Yearly Expenses, and BSR
    sales = df.loc["Sales"].astype(float)  # Replace with actual 'Sales' row from df
    yearly_expenses = df.loc["YearlyExpenses"].astype(float)  # Replace with actual 'YearlyExpenses' row from df
    bsr = df.loc["BSR"].astype(float)  # Replace with actual 'BSR' row from df

    # Ensure that the length of the report_dates matches the trends data
    if len(report_dates) != len(sales) or len(report_dates) != len(yearly_expenses) or len(report_dates) != len(bsr):
        raise ValueError("The length of Report Date and trend data do not match.")

    # Create the plot
    plt.figure(figsize=(12, 7))

    # Plot the three trends with brighter colors
    plt.plot(report_dates, sales, label="Sales", color='#00bfae', linewidth=3)  # Bright cyan-like color
    plt.plot(report_dates, yearly_expenses, label="Yearly Expenses", color='#ff4081', linewidth=3)  # Bright pink
    plt.plot(report_dates, bsr, label="BSR", color='#ffeb3b', linewidth=3)  # Bright yellow

    # Customize font properties for a modern look
    plt.title("Growth Trends: Sales & BSR", fontsize=18, color='white', fontweight='bold')
    plt.xlabel("Year", fontsize=14, color='white')
    plt.ylabel("Values", fontsize=14, color='white')

    # Set legend with better styling
    plt.legend(facecolor='white',loc='upper left', fontsize=12, labelcolor='white', frameon=False, labels=['Sales', 'Yearly Expenses', 'BSR'])

    # Set grid lines with white and more contrast
    plt.grid(True, which='both', axis='y', linestyle='-', color='white', alpha=0.3)

    # Set figure and axes backgrounds to transparent
    plt.gca().patch.set_facecolor('none')  # Transparent axes background
    plt.gcf().patch.set_facecolor('none')  # Transparent figure background

    # Apply white color to ticks for a clean look
    plt.tick_params(axis='both', labelsize=12, labelcolor='white', colors='white')

    # Show the plot in Streamlit
    st.pyplot(plt)
    '''


    "---"


    

"---"
st.write("""
         Complete DATA for Further Analysis
         """)






