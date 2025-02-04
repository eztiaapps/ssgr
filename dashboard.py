import logging
from PIL import Image, ImageEnhance
import time
import json
import requests
import base64
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from splfunction import *


# Streamlit Page Configuration
st.set_page_config(
    page_title="Streamly - An Intelligent Streamlit Assistant",
    page_icon="imgs/avatar_streamly.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/AdieLaine/Streamly",
        "Report a bug": "https://github.com/AdieLaine/Streamly",
        "About": """
            ## Streamly Streamlit Assistant
            ### Powered using GPT-4o-mini

            **GitHub**: https://github.com/AdieLaine/

            The AI Assistant named, Streamly, aims to provide the latest updates from Streamlit,
            generate code snippets for Streamlit widgets,
            and answer questions about Streamlit's latest features, issues, and more.
            Streamly has been trained on the latest Streamlit updates and documentation.
        """
    }
)

# Function to reset session state when file is removed
def reset_state():
    st.session_state.uploaded_file = None
    st.session_state.df = None  # Reset dataframe


def display_disclaimer():
    """Display the latest disclaimer of the Portfolio Assistant."""
    with st.expander("Portfolio Assistant 1.0 Announcement", expanded=False):
        st.markdown("For more details on this version, check out the [Disclaimer](https://github.com/eztiaapps/ssgr/blob/main/disclaimer.txt).")

# Streamlit Title
st.title("Datalotus Portfolio Assistant")

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None


def main():
    """
    Analyses a stock and provides buy and signal weighted factors for decision making.
    """

    # Insert custom CSS for glowing effect
    st.markdown(
        """
        <style>
        .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #1b4f59,   /* Deep teal */
            0 0 10px #206972,  /* Dark cyan */
            0 0 15px #268b96,  /* Theme-aligned blue-green */
            0 0 20px #3aa8b8,  /* Vibrant cyan */
            0 0 25px #5ec4d1,  /* Softened teal */
            0 0 30px #7fdde7,  /* Lighter aqua */
            0 0 35px #a0f0ff;
        position: relative;
        z-index: -1;
        border-radius: 45px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Load and display sidebar image
    img_path = "imgs/stuser.jpeg"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/jpeg;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    # Sidebar for Mode Selection
    source = st.sidebar.radio("Stock Balance Sheet Source:", options=["Screener", "Trendlyne", "MoneyControl"], index=0)

    st.sidebar.markdown("---")

    # Display basic interactions
    show_basic_info = st.sidebar.checkbox("Show Basic Interactions", value=True)
    if show_basic_info:
        st.sidebar.markdown("""
        ### Basic Interactions
        - **Find a stock balance sheet :** Search and get a balance sheet from (e.g. Screener.in) in excel format.
        - **Upload it in the main screen**: Simply upload.
        - **AI Model**: Our AI Model will analyse all business numbers and share the result in the screen.
        """)

    # Display advanced interactions
    show_advanced_info = st.sidebar.checkbox("Show Advanced Interactions", value=False)
    if show_advanced_info:
        st.sidebar.markdown("""
        ### Advanced Interactions
        - **Work in Progress** - This site is still under construction!
        - **Generate Report**: In upcoming release you will be able to download the report in PDFs.
        - **Multiple File Upload**: You will be able to upload multiple files simultaneously for quick portfolio analysis.
        - **Connect with our CAs**: You can book 1:1 session with our certified CAs for your Portfolio Analysis.
        
        """)

    st.sidebar.markdown("---")

    with st.expander(("About Datalotus Portfolio Assistant")):

        st.markdown((
            """
        The **#Datalotus Portfolio Assistant** is an AI powered portfolio analysis tool.
        Like a lotus blooms in murky water, 1 really **good stock** can be mixed with 99 other **good looking** stocks.!
        
        ### Particularly, we use this tool to:
        - Avoid bad investments! No investment is better than speculating or jump on rallies or catching falling knives!
        - Do our own stock's fundamental analysis
        - Focus on business performance rather than market speculation or sentiments.
        - Learn about all the correct fundamentally strong stocks with simple indicators.
        - We will display for green and red flags based on the stock's balance sheet uploaded below.
        - Please read our [Disclaimer](https://github.com/eztiaapps/ssgr/blob/main/disclaimer.txt).
        """
        ))

    

       

    if source == "Screener":
        st.header("Welcome to the world of possibilities!")
        "---"

        st.write("""

        #### Upload the stock's balance sheet 📈,
         
         """)
        
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
        



        
    else:
        st.markdown("""
                    This part of the site is under construction!
                    In the meanwhile, you can read our Disclaimer to use this site!
                    """)
        display_disclaimer()
        

    
if __name__ == "__main__":
    main()