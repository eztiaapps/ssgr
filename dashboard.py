import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from splfunction import read_process_excel
import matplotlib.pyplot as plt
import seaborn as sns


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
         ### ...Coming soon!
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

# Collects the stock csv file from user
uploaded_file = st.sidebar.file_uploader("Upload that stock balance sheet here", type=["xlsx"])
if uploaded_file is not None:
    df = read_process_excel(uploaded_file)
    file_name =  uploaded_file.name.split('.')[0]

    st.write("1. Checkpoint: Did Expense increase with Sales?")
    st.write(f"""
             # Sales vs Yearly Expenses... {file_name}
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
        sns.lineplot(x=report_dates, y=sales_data, marker="o", label="Sales", ax=ax)
        sns.lineplot(x=report_dates, y=expenses_data, marker="s", label="YearlyExpenses", ax=ax)

        ax.set_title("Sales vs Yearly Expenses Trend", fontsize=14)
        ax.set_xlabel("Report Date", fontsize=12)
        ax.set_ylabel("Amount", fontsize=12)
        ax.legend()
        # **Lighter Grid**
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.2, color="white")  # Fainter grid lines

        # Remove the white background
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Show the chart
        st.pyplot(fig)
    else:
        st.error("Sales or YearlyExpenses data is missing!")
    
    
            


"---"



