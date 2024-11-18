import streamlit as st
from streamlit_ace import st_ace
import re
import pandas as pd

# Set up the title of the web app
st.title('Live Text Editor with Word Count')

# Create columns layout with better proportions
col1, col2 = st.columns([3, 1])

# Create an Ace editor for inputting and editing text in the left column
with col1:
    text_input = st_ace(language='text', theme='monokai', height=800, auto_update=True)

    # Calculate the word count dynamically
    word_count = len(text_input.split()) if text_input.strip() else 0

    # Display the word count below the editor for better visibility
    st.markdown(f"<div style='padding: 10px; font-size: 18px;'>Word count: <strong>{word_count}</strong></div>", unsafe_allow_html=True)

# Load list of words to check from CSV file
try:
    words_df = pd.read_csv('important_words.csv')
    words_to_check = words_df[['words', 'minimal frequency']].to_dict('records')
except FileNotFoundError:
    st.sidebar.error("The file 'important_words.csv' was not found.")
    words_to_check = []

# Check word occurrences in the text
word_counts = {word['words']: len(re.findall(r'\b' + re.escape(word['words']) + r'\b', text_input)) for word in words_to_check}

# Display word usage in the sidebar with better styling
with st.sidebar:
    st.markdown("<div style='text-align: center; font-size: 24px; font-weight: bold;'>Word Usage</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding: 10px; background-color: #f0f0f0; border-radius: 10px;'>", unsafe_allow_html=True)
    for word_info in words_to_check:
        word = word_info['words']
        minimal_frequency = word_info['minimal frequency']
        count = word_counts[word]
        if count > minimal_frequency * 2:
            color = "#ff6666"  # Red color if count exceeds twice the minimal frequency
        elif count >= minimal_frequency:
            color = "#8ef"  # Light green if count meets minimal frequency
        else:
            color = "#faa"  # Light pink if count does not meet minimal frequency        
        st.markdown(f"<div style='display: flex; justify-content: space-between; margin-bottom: 5px; padding: 8px; background-color: {color}; color: black; border-radius: 5px;'>"
                    f"<span style='font-weight: bold;'>{word}</span>"
                    f"<span>{count}/{minimal_frequency} occurrences</span>"
                    f"</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

