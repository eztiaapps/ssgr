import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from splfunction import read_process_excel
import matplotlib.pyplot as plt
import seaborn as sns


input_df = pd.DataFrame()


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
         
         _This app calculates the safety margin, evaluate key business numbers and helps to avoid bad investments._
         ### ...Under Construction...!
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

    st.write("1. Checkpoint: Did Expense increase with Sales?")
    st.write(f"""
             # Sales vs Yearly Expenses : {file_name}
             """)
    
    # Extract sales and expenses data
    if "Sales" in df.index and "YearlyExpenses" in df.index:
        sales_data = df.loc["Sales"].astype(float)
        expenses_data = df.loc["YearlyExpenses"].astype(float)
        report_dates = df.loc["Report Date"].tolist()  # Extract report dates

        # Convert report dates to actual datetime
        report_dates = pd.to_datetime(report_dates, errors="coerce")

        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.05)  # Make figure background transparent
        ax.set_facecolor("none")  # Make plot area transparent

        # White legend text
        legend = ax.legend(facecolor="none")
        for text in legend.get_texts():
            text.set_color("white")


        sns.lineplot(x=report_dates, y=sales_data, marker="o", label="Sales", ax=ax, color="blue")
        sns.lineplot(x=report_dates, y=expenses_data, marker="s", label="YearlyExpenses", ax=ax, color="red")

        ax.set_title("Sales vs Yearly Expenses Trend", fontsize=14, color="white")
        ax.set_xlabel("Report Date", fontsize=12, color="white")
        ax.set_ylabel("Amount (Cr.)", fontsize=12, color="white")
        ax.legend()
        # **Lighter Grid**
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.2, color="white")  # Fainter grid lines

        # Remove the white background
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set white color for X and Y ticks
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Show the chart
        st.pyplot(fig)

    else:
        st.error("Sales or YearlyExpenses data is missing!")

                
    "---"
    st.write(f"""
             # Sales vs Expenses vs Operating Profit
             """)

    # **Step 1: Calculate Operating Profit**
    if "Sales" in df.index and "YearlyExpenses" in df.index:
        df.loc["Operating Profit"] = df.loc["Sales"].astype(float) - df.loc["YearlyExpenses"].astype(float)

    # **Step 2: Extract data for plotting**
        if all(key in df.index for key in ["Sales", "YearlyExpenses", "Operating Profit"]):
            sales_data = df.loc["Sales"].astype(float)
            expenses_data = df.loc["YearlyExpenses"].astype(float)
            profit_data = df.loc["Operating Profit"].astype(float)
            report_dates = pd.to_datetime(df.loc["Report Date"].tolist(), errors="coerce")

            # **Step 3: Plot the data**
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_alpha(0)  # Transparent figure background
            ax.set_facecolor("none")  # Transparent plot area
            
            # White legend text
            legend = ax.legend(facecolor="none")
            for text in legend.get_texts():
                text.set_color("white")


            # Plot Sales, Expenses, and Profit
            sns.lineplot(x=report_dates, y=sales_data, marker="o", label="📈 Sales", ax=ax, color="blue")
            sns.lineplot(x=report_dates, y=expenses_data, marker="s", label="💰 YearlyExpenses", ax=ax, color="red")
            sns.lineplot(x=report_dates, y=profit_data, marker="^", label="🚀 Operating Profit", ax=ax, color="green")

            ax.set_title("Sales, Expenses & Operating Profit Trend", fontsize=14, color="white")
            ax.set_xlabel("Report Date", fontsize=12, color="white")
            ax.set_ylabel("Amount(Cr.)", fontsize=12, color="white")
            ax.legend()  # Transparent legend

            # Lighter Grid
            ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.2, color="white")

            # Remove Borders
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Set white color for X and Y ticks
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')

            # Show the transparent chart
            st.pyplot(fig)

            # Mark chart as drawn
            st.session_state.chart_drawn = True
        else:
            st.error("Sales, YearlyExpenses, or Operating Profit data is missing!")

    
    # **Step 1: Calculate Operating Profit Margin**
    if "Sales" in df.index and "Operating Profit" in df.index:
        df.loc["OPM%"] = df.loc["Operating Profit"].astype(float) / df.loc["Sales"].astype(float)
    
    else:
        st.warning("⚠️ 'Calculation has missing inputs")
    
    # Calculate NPM% (Net Profit Margin)
    # Ensure the values in 'Net Profit' and 'Sales' are numeric
    net_profit = pd.to_numeric(df.loc["Net profit"], errors='coerce')
    sales = pd.to_numeric(df.loc["Sales"], errors='coerce')

    # Calculate NPM% (Net Profit Margin)
    if not pd.isna(net_profit) and not pd.isna(sales) and sales != 0:
        df.loc["NPM%"] = (net_profit / sales) * 100
    else:
        st.warning("⚠️ 'Net Profit' or 'Sales' row is invalid. NPM% cannot be calculated.")

    

    #Copy the final data to global dataframe
    df.index = df.index.astype(str)
    input_df = df.copy()
    



"---"
st.write("""
         Complete DATA for Further Analysis
         """)
st.write(input_df)





