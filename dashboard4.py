import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from splfunction import read_process_excel, plot_growth_vs_bsr
import matplotlib.pyplot as plt
import seaborn as sns





#---------------Settings ----------------------#
currency = "INR"
page_title = "DataLotus Dashboard"
page_icon= ":money_with_wings:"
layout = "centered"

#---------------Settings ----------------------#

# Custom HTML to center title
st.markdown(
    """
    <h1 style='text-align: center;'>Datalotus Portfolio Analysis</h1>
    """,
    unsafe_allow_html=True
)

#st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
#st.title(page_title + " " + page_icon)



st.write("""

        ## Welcome to AI based Stock Analysis,
         
         """)



# Function to reset session state when file is removed
def reset_state():
    st.session_state.uploaded_file = None
    st.session_state.df = None  # Reset dataframe

st.write("""

        ### Upload the stock balance sheet,
         
         """)
st.header("Upload Stock Balance Sheet 📈")
"---"

# Sidebar file uploader
uploaded_file = st.file_uploader("Upload that stock balance sheet here", type=["xlsx"])

# Check if file was removed or changed
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "df" not in st.session_state:
    st.session_state.df = None  # Initialize empty

# If file is removed, reset session state and force rerun
if uploaded_file is None and st.session_state.uploaded_file is not None:
    reset_state()
    st.rerun()  # Forces Streamlit to refresh the UI

# If a new file is uploaded, process it
if uploaded_file is not None and uploaded_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = uploaded_file  # Store new file
    st.session_state.df = read_process_excel(uploaded_file)  # Process Excel file

# Display checkpoint info if DataFrame exists
if st.session_state.df is not None:
    df = st.session_state.df
    file_name = uploaded_file.name.split('.')[0]

    st.write("1. Checkpoint: Is Sales higher than BSR?")
    st.write(f"# Sales vs Business Sustainability: {file_name}")

    # Display DataFrame
    st.write(df)
    "---"

    # Call the plotting function
    plot_growth_vs_bsr(df)








