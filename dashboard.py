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
from splfunction import read_process_excel, plot_growth_vs_bsr
import matplotlib.pyplot as plt
import seaborn as sns


def display_disclaimer():
    """Display the latest disclaimer of the Portfolio Assistant."""
    with st.expander("Portfolio Assistant 1.0 Announcement", expanded=False):
        st.markdown("For more details on this version, check out the [Disclaimer](https://docs.streamlit.io/library/changelog#version).")

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

    if source == "Screener":
        st.header("Welcome")
    else:
        display_disclaimer()
        

    
if __name__ == "__main__":
    main()