import streamlit as st
from streamlit_ace import st_ace
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import spacy
import math
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import time
from urllib.parse import urlparse
import altair as alt

# Load SpaCy model for POS tagging
nlp = spacy.load('en_core_web_sm')

# Function to extract content from a URL with retries and user-agent header
def extract_content_from_url(url, retries=2, timeout=5):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                if content.strip():
                    return content
        except requests.RequestException as e:
            print(f"Error fetching content from {url}: {e}. Attempt {attempt + 1} of {retries}.")
        time.sleep(2)  # Wait before retrying
    return ""

# Function to get top 10 Google SERP results for a keyword (unique domains)
def get_top_10_unique_domain_results(keyword):
    try:
        results = []
        domains = set()
        for url in search(keyword, num_results=20):
            domain = urlparse(url).netloc
            if domain not in domains:
                domains.add(domain)
                results.append(url)
            if len(results) == 2:
                break
        return results
    except Exception as e:
        st.error(f"Error during Google search: {e}")
        return []

# Function to filter out stopwords, auxiliary verbs, and other less informative words
def filter_terms(terms):
    filtered_terms = []
    for term in terms:
        doc = nlp(term)
        if not doc[0].is_stop and doc[0].pos_ not in ['AUX', 'PRON', 'DET', 'ADP', 'CCONJ']:
            filtered_terms.append(term)
    return filtered_terms

# Streamlit App
def main():
    st.title('Needs More Words! Optimize Your Content')

    # Session state to keep track of analysis stage
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False

    # Ask the user to input a keyword
    if 'editor_started' not in st.session_state or not st.session_state.editor_started:
        keyword = st.text_input('Enter a keyword to retrieve content:')
        start_analysis = st.button('Start Analysis')
    else:
        keyword = ''
        start_analysis = False

    if keyword and start_analysis:
        print("Start Analysis button clicked.")
        # Get top 10 search results for the keyword
        st.info('Retrieving top 10 search results...')
        top_urls = get_top_10_unique_domain_results(keyword)
        print(f"Top URLs retrieved: {top_urls}")

        if top_urls:
            retrieved_content = []
            for idx, url in enumerate(top_urls):
                status_message = st.empty()
                try:
                    status_message.text(f"Now trying to retrieve content from {url}")
                    content = extract_content_from_url(url)
                    if content:
                        retrieved_content.append(content)
                        status_message.text(f"Successfully retrieved content from {url}")
                    else:
                        status_message.text(f"No meaningful content found at {url}")
                    print(f"Content retrieved from {url}: {content[:100]}...")
                except ValueError as e:
                    status_message.text(f"Error: {str(e)} from {url}")
                time.sleep(1)

            # Proceed only if content has been retrieved
            if retrieved_content:
                documents = retrieved_content
                print("Documents retrieved successfully.")

                

                # Initialize TF and TF-IDF Vectorizers with n-grams (uni-, bi-, tri-grams)
                tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
                tf_vectorizer = CountVectorizer(ngram_range=(1, 3))

                # Fit the model and transform the documents into TF and TF-IDF matrices
                tfidf_matrix = tfidf_vectorizer.fit_transform(documents).toarray()
                tf_matrix = tf_vectorizer.fit_transform(documents).toarray()

                # Extract feature names (terms)
                feature_names = tfidf_vectorizer.get_feature_names_out()
                print(f"Feature names extracted: {feature_names[:10]}...")

                # Filter feature names to exclude less informative words
                filtered_feature_names = filter_terms(feature_names)
                print(f"Filtered feature names: {filtered_feature_names[:10]}...")

                # Filter TF-IDF and TF matrices to only include filtered terms
                filtered_indices = [i for i, term in enumerate(feature_names) if term in filtered_feature_names]
                tfidf_matrix_filtered = tfidf_matrix[:, filtered_indices]
                tf_matrix_filtered = tf_matrix[:, filtered_indices]

                # Calculate average TF-IDF and TF scores
                avg_tfidf_scores = np.mean(tfidf_matrix_filtered, axis=0)
                avg_tf_scores = np.mean(tf_matrix_filtered, axis=0)
                print(f"Average TF-IDF scores calculated: {avg_tfidf_scores[:10]}...")

                # Multiply TF-IDF scores by 1000 for better visualization
                avg_tfidf_scores_scaled = [score * 1000 for score in avg_tfidf_scores]

                # Get top 50 terms by average TF-IDF score
                sorted_indices = np.argsort(avg_tfidf_scores)[-50:][::-1]
                highlighted_feature_names = [filtered_feature_names[i] for i in sorted_indices]
                highlighted_avg_tfidf_scores = [avg_tfidf_scores_scaled[i] for i in sorted_indices]
                highlighted_avg_tf_scores = [avg_tf_scores[i] for i in sorted_indices]
                print(f"Top 50 terms: {highlighted_feature_names[:10]}...")

                # Plot the top 50 terms by average TF and TF-IDF scores using a stacked bar chart
                chart_data = pd.DataFrame({
                    'Terms': highlighted_feature_names,
                    'TF Score': highlighted_avg_tf_scores,
                    'TF-IDF Score (scaled)': highlighted_avg_tfidf_scores
                }).set_index('Terms')

                # Plot the top 50 terms by average TF and TF-IDF scores using a dynamic bar chart
                st.subheader('Top 50 Words TF & TF-IDF Scores')
                st.bar_chart(chart_data, color=["#FFAA00", "#6495ED"])



                # Store the results in session state for the next stage
                st.session_state.chart_data = chart_data
                
                st.session_state.words_to_check = [
                    {
                        'Term': highlighted_feature_names[i],
                        'Average TF Score': highlighted_avg_tf_scores[i],
                        'Average TF-IDF Score (scaled)': highlighted_avg_tfidf_scores[i]
                    }
                    for i in range(len(highlighted_feature_names))
                ]
                st.session_state['avg_tf_scores'] = highlighted_avg_tf_scores
                st.session_state.analysis_completed = True
                

    # Continue to editor button
    if 'editor_started' not in st.session_state or not st.session_state.editor_started:    
        if st.session_state.analysis_completed:
            continue_to_editor = st.button('Continue to Editor')
            if continue_to_editor:
                st.session_state.editor_started = True        

    # Proceed to editor once the user clicks the continue button
    if 'editor_started' in st.session_state and st.session_state.editor_started:
        print("Proceeding to the st-ace editor and sidebar stage.")
        # Add a button to start a new analysis
        if st.button('Start a New Analysis'):
            st.session_state.clear()
            st.stop()

        st.sidebar.subheader('Kepp an eye on these')
        words_to_check = st.session_state.words_to_check

        # Create columns layout with better proportions
        col1, col2 = st.columns([3, 1])

        # Create an Ace editor for inputting and editing text in the left column
        with col1:
            text_input = st_ace(language='text', theme='monokai', height=400, auto_update=True, show_gutter=False)

            # Calculate the word count dynamically
            word_count = len(text_input.split()) if text_input.strip() else 0

            # Display the word count below the editor for better visibility
            st.markdown(f"<div style='padding: 10px; font-size: 18px;'>Word count: <strong>{word_count}</strong></div>", unsafe_allow_html=True)

            #redraw the chart
            if 'chart_data' in st.session_state:
                st.subheader('Comparison of Average TF and Editor TF Scores')

                # Calculate TF-IDF scores for each term in the editor content
                if text_input.strip():
                    total_words = len(text_input.split())
                    tf_vectorizer2 = CountVectorizer(vocabulary=[word['Term'] for word in words_to_check], ngram_range=(1, 3))
                    text_tf_matrix = tf_vectorizer2.fit_transform([text_input]).toarray()
                    editor_tf_scores = (text_tf_matrix[0] / total_words) * 1000  # Calculate TF as term frequency divided by total words and scale for visualization

                    # Create DataFrame for comparison chart
                    comparison_chart_data = pd.DataFrame({
                        'Terms': [word['Term'] for word in words_to_check],
                        'Average TF Score': [word['Average TF Score'] for word in words_to_check],
                        'Editor TF Score': editor_tf_scores
                    })

                    # Create Altair chart with bars and lines
                    base = alt.Chart(comparison_chart_data).encode(x=alt.X('Terms:N', sort='-y', title='Terms'))

                    bar = base.mark_bar(color='steelblue').encode(
                        y=alt.Y('Average TF Score:Q', title='TF Scores')
                    )

                    line = base.mark_line(color='orange').encode(
                        y=alt.Y('Editor TF Score:Q')
                    )

                    combined_chart = (bar + line).properties(width=600, height=400)
                    st.altair_chart(combined_chart, use_container_width=True)
                    
        # Calculate TF scores for each term in the sidebar based on the editor content
        if text_input.strip():
            total_words = len(text_input.split())
            text_vectorizer = CountVectorizer(vocabulary=[word['Term'] for word in words_to_check], ngram_range=(1, 3))
            text_matrix = text_vectorizer.fit_transform([text_input]).toarray()
            text_tf_scores = text_matrix[0] / total_words * 1000 # Calculate TF as term frequency divided by total words

            # Display word usage in the sidebar with better styling
            with st.sidebar:
                st.markdown("<div style='text-align: center; font-size: 24px; font-weight: bold;'>Words</div>", unsafe_allow_html=True)
                st.markdown("<div style='padding: 10px; background-color: #f0f0f0; border-radius: 10px;'>", unsafe_allow_html=True)
                for idx, word_info in enumerate(words_to_check):
                    word = word_info['Term']
                    tf_score = text_tf_scores[idx]
                    occurrences = len(re.findall(r'\b' + re.escape(word) + r'\b', text_input, flags=re.IGNORECASE))
                    
                    avg_tf_score = word_info['Average TF Score']
                    target = math.floor(avg_tf_score * word_count / 1000)

                    if tf_score >= avg_tf_score:
                        color = "#b0e57c"  # Light green if the term frequency meets or exceeds the average TF score
                    else:
                        color = "#ffcccb"  # Light pink if the term frequency is below the average TF score
                    st.markdown(f"<div style='display: flex; justify-content: space-between; margin-bottom: 5px; padding: 8px; background-color: {color}; color: black; border-radius: 5px;'>"
                                f"<span style='font-weight: bold;'>{word}</span>"
                                f"<span>{occurrences} / {target} Occurrences</span>"
                                f"</div>", unsafe_allow_html=True)
                    progress = min(1.0, occurrences / target) if target > 0 else 0
                    st.progress(progress)
                    
                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
