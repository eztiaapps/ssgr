import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from splfunction import split_excel_into_dataframes, pnlpreprocess
import matplotlib.pyplot as plt


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
    input_df = split_excel_into_dataframes(uploaded_file)
    file_name =  uploaded_file.name
    pnl = pnlpreprocess(input_df['PROFIT & LOSS'])
    print(type(file_name))

    st.markdown(f"### Yearly Sales Growth Trend - {file_name}")
    st.write(pnl)

    # ✅ Transpose DataFrame
    df = pnl.loc[0]
    df = df.to_frame().T
    df = df.set_index("Report Date").T.reset_index()

    # ✅ Rename Columns
    df.columns = ["Report Date", "Sales"]

    # ✅ Convert Data Types
    df["Report Date"] = pd.to_datetime(df["Report Date"])  # Convert to DateTime
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")  # Convert to Float
    #✅ Calculate Sales Growth (%)
    df["Sales Growth (%)"] = df["Sales"].pct_change() * 100

    # ✅ Display Result
    st.write("""
         ### 1. Checkpoint : whether Sales growth increasing or decreasing!
         """)

    st.write(df)
    
    

    # Create the Streamlit app title
    st.title("📈 Sales Growth Trend")

    # Create a polished chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the Sales Growth (%) trend
    ax.plot(df["Report Date"], df["Sales Growth (%)"], marker="o", linewidth=2.5,
            color="#1f77b4", label="Sales Growth (%)")

    # Add a horizontal line at zero for reference
    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.8)

    # Set axis labels and title with modern font sizes
    ax.set_title("Yearly Sales Growth Trend", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=14, labelpad=10)
    ax.set_ylabel("Sales Growth (%)", fontsize=14, labelpad=10)

    # Format the x-axis dates
    fig.autofmt_xdate(rotation=45)

    # Add grid lines for readability (already present with seaborn-whitegrid, but you can customize)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Optionally, add annotations for each data point (if desired)
    for x, y in zip(df["Report Date"], df["Sales Growth (%)"]):
        if pd.notnull(y):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=10, color="#555555")

    # Add a legend with a modern frame
    ax.legend(frameon=True, framealpha=0.9, edgecolor="gray", fontsize=12)

    # Tight layout for better spacing
    fig.tight_layout()

    # Display the polished chart in Streamlit
    st.pyplot(fig)
    
            


"---"



