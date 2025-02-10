import logging
from PIL import Image, ImageEnhance
import time
import json
import requests
import base64
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from splfunction import *
from metrics import POSITIVE_METRICS, NEGATIVE_METRICS



# Streamlit Page Configuration
st.set_page_config(
    page_title="Datalotus - An Intelligent AI Based Portfolio Assistant",
    page_icon="imgs/slogo.png",
    layout="centered",
    initial_sidebar_state="auto",
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
    
    # Streamlit Title
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.header(f"Datalotus Portfolio Assistant")

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
        st.markdown(
            "<h1 style='text-align: center; font-family: Times New Roman, serif; color: cyan;'>AI Powered Fundamental Analysis!</h1>",
            unsafe_allow_html=True
        )   
        "---"

        st.markdown(
            f"<h3 style='text-align: center; font-family: Times New Roman, serif;'>Upload the stock's balance sheet 📈 from {source}.in </h3>",
            unsafe_allow_html=True
        )
        

        # Sidebar file uploader
        uploaded_file = st.file_uploader("Please upload here:", type=["xlsx"])
        # Increment and display the visitor count
        visitor_count = get_visitor_count()

        st.write(f"Visitors have used Datalotus AI Assistant: {visitor_count} times, so far this week!")

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
            visitor_count = increment_visitor_count()
            
        
        # Display checkpoint info if DataFrame exists
        
        if st.session_state.df is not None:
            df = st.session_state.df
            file_name = uploaded_file.name.split('.')[0].upper()

            st.markdown(
                f"<h3 style='text-align: center; font-family: Times New Roman, serif;'>Let's Score This Business: {file_name}</h3>",
                unsafe_allow_html=True
            )
            "---"
            #st.write(df)  #Comment Uncomment to show raw data for testing purpose.
            # Assume df is already loaded and processed
            
            if "Report Date" in df.index:
                st.markdown(
                    "<h3 style='text-align: center; font-family: Times New Roman, serif;'>Individual Growth Scores</h3>",
                    unsafe_allow_html=True
                )
                all_scores = []

                # Manually call the function for each metric
                for metric in POSITIVE_METRICS + NEGATIVE_METRICS:
                    st.subheader(f"{metric} Growth Score")
                    score_5_years = display_score(metric, df, 5)
                    score_3_years = display_score(metric, df, 3)

                    # Collect scores for overall calculation
                    if score_5_years is not None:
                        all_scores.append(score_5_years)
                    if score_3_years is not None:
                        all_scores.append(score_3_years)

                # Calculate overall score
                overall_score = calculate_overall_score(all_scores)

                if overall_score is not None:
                    st.subheader("👮🏻‍♂️ Overall Growth Score")
                    st.slider("Overall Growth Score", 0, 100, int(overall_score), disabled=True, format="%d")
                    


                    # Assign color based on score
                    if overall_score < 34:
                        color = "red"
                        label = "Poor"
                    elif overall_score < 67:
                        color = "orange"
                        label = "Average"
                    else:
                        color = "green"
                        label = "Good"
                    
                    st.markdown(colored_text(f"Overall Score is: {overall_score}, for {file_name}", 'Red'), unsafe_allow_html=True)
                    st.markdown(f"<span style='color:{color}; font-size:22px; font-weight:bold'>The score is: {label}</span>", unsafe_allow_html=True)
                    
                    st.markdown(colored_text(f"It must be Good, only then we can consider a company for investment!", 'orange'), unsafe_allow_html=True)
                    "---"
                    st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical space

                else:
                    st.warning("Not enough data for overall score calculation.")

            else:
                st.warning("Report Date data not found in DataFrame.")

            # Display DataFrame
            # TODO: Intrinsic Value and Safety Margin
            #ReportSummary(df)
            
            st.write ("# 👮🏻‍♂️ Target Price or Fair Value, Safety Margin")
            
            #Let's calculate the Fair value for 3 and 5 years historical.
            # Get PE and EPS values for last 3 years
            pe_values_3y = get_metric_values_last_n_years(df, 'PE', 3)
            eps_values_3y = get_metric_values_last_n_years(df, 'EPS', 3)

            # Get PE and EPS values for last 5 years
            pe_values_5y = get_metric_values_last_n_years(df, 'PE', 5)
            eps_values_5y = get_metric_values_last_n_years(df, 'EPS', 5)

            print("\n" + 'pe and eps values')
            print(pe_values_3y)
            print(pe_values_5y)
            print(eps_values_3y)
            print(eps_values_5y)

            # Calculate CAGRs for PE and EPS for 3 and 5 years
            pe_cagr_3y = calculate_cagr(pe_values_3y)
            pe_cagr_5y = calculate_cagr(pe_values_5y)
            eps_cagr_3y = calculate_cagr(eps_values_3y)  
            eps_cagr_5y = calculate_cagr(eps_values_5y)  

            # Get current PE and EPS
            pe_current = df.loc["PE"].dropna().iloc[-1]
            eps_current = df.loc["EPS"].dropna().iloc[-1]  

            # Calculate fair value for 5 years
            fair_value_5y = (1 + pe_cagr_5y) * pe_current * (1 + eps_cagr_5y) * eps_current
            
            st.subheader("**Based on past 5 years performance,**")
            st.write(f"Fair Value or Future Value should be: {fair_value_5y:.2f} INR")

            # Calculate fair value for 3 years
            fair_value_3y = (1 + pe_cagr_3y) * pe_current * (1 + eps_cagr_3y) * eps_current
            
            st.subheader("**Based on past 3 years performance,**")
            st.write(f"Fair Value or Future Value should be: {fair_value_3y: .2f} INR")

            st.warning("Overall Growth Score should be: Good and only then we use Target Value! Else Avoid!")

            "---"

            st.markdown(
            """
            <div style="text-align: center; font-family: Times New Roman, serif;">
                <h2 style="font-size: 26px; font-family: Times New Roman, serif;">Let's understand the result?</h2>
                <h4 style="text-align: left; font-family: Times New Roman, serif; color: yellow">These 3 Signals are must!</h4>
                <ul style="text-align: left;">
                    <li><b>1. Check whether BSR is good or average, It must be Good :!</b></li>
                    <li><b>2. Overall Growth score must be Good!</b></li>
                    <li><b>3. Target Price should have Safety Margin of at least 30%. [...Coming soon!]</b></li>
                </ul>
                
            </div>
            <p>If any of the above 3 is missing, we avoid that stock, & choose another one!</p>
            """,
            unsafe_allow_html=True
        )

            


        
    else:
        st.markdown("""
                    This part of the site is under construction!
                    In the meanwhile, you can read our Disclaimer to use this site!
                    """)
        display_disclaimer()








    
if __name__ == "__main__":
    main()
    

    