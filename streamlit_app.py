import re
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import streamlit as st
from streamlit_quill import st_quill  # Import the streamlit-quill component
from googlesearch import search
import altair as alt

# Load SpaCy model for POS tagging with caching
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        from spacy.cli import download
        download('en_core_web_sm')
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

# Function to extract content from a URL with retries and user-agent header
@st.cache_data
def extract_content_from_url(url, retries=2, timeout=5):
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        )
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                if content.strip():
                    return content
            else:
                st.warning(f"Received status code {response.status_code} from {url}")
        except requests.RequestException as e:
            st.warning(f"Error fetching content from {url}: {e}. Attempt {attempt + 1} of {retries}.")
        time.sleep(2)  # Wait before retrying
    return ""

# Function to get top 10 Google SERP results for a keyword (unique domains)
@st.cache_data
def get_top_10_unique_domain_results(keyword):
    try:
        results = []
        domains = set()
        for url in search(keyword, num_results=20):
            domain = urlparse(url).netloc
            if domain not in domains:
                domains.add(domain)
                results.append(url)
            if len(results) == 10:
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

def perform_analysis(keyword):
    st.info('Retrieving top 10 search results...')
    top_urls = get_top_10_unique_domain_results(keyword)
    if not top_urls:
        st.error('No results found.')
        return

    retrieved_content = []
    for idx, url in enumerate(top_urls):
        status_message = st.empty()
        status_message.text(f"Retrieving content from {url}...")
        content = extract_content_from_url(url)
        if content:
            retrieved_content.append(content)
            status_message.text(f"Successfully retrieved content from {url}")
        else:
            status_message.text(f"No meaningful content found at {url}")
        time.sleep(1)

    if not retrieved_content:
        st.error('Failed to retrieve content from the URLs.')
        return

    documents = retrieved_content

    # Initialize TF and TF-IDF Vectorizers with n-grams (uni-, bi-, tri-grams)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tf_vectorizer = CountVectorizer(ngram_range=(1, 3))

    # Fit the model and transform the documents into TF and TF-IDF matrices
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents).toarray()
    tf_matrix = tf_vectorizer.fit_transform(documents).toarray()

    # Extract feature names (terms)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Filter feature names to exclude less informative words
    filtered_feature_names = filter_terms(feature_names)

    # Filter TF-IDF and TF matrices to only include filtered terms
    filtered_indices = [i for i, term in enumerate(feature_names) if term in filtered_feature_names]
    tfidf_matrix_filtered = tfidf_matrix[:, filtered_indices]
    tf_matrix_filtered = tf_matrix[:, filtered_indices]

    # Calculate average TF-IDF and TF scores
    avg_tfidf_scores = np.mean(tfidf_matrix_filtered, axis=0)
    avg_tf_scores = np.mean(tf_matrix_filtered, axis=0)

    # Multiply TF-IDF scores by 1000 for better visualization
    avg_tfidf_scores_scaled = avg_tfidf_scores * 1000

    # Get top 50 terms by average TF-IDF score
    sorted_indices = np.argsort(avg_tfidf_scores)[-50:][::-1]
    highlighted_feature_names = [filtered_feature_names[i] for i in sorted_indices]
    highlighted_avg_tfidf_scores = [avg_tfidf_scores_scaled[i] for i in sorted_indices]
    highlighted_avg_tf_scores = [avg_tf_scores[i] for i in sorted_indices]

    # Store the results in session state
    st.session_state['chart_data'] = pd.DataFrame({
        'Terms': highlighted_feature_names,
        'Average TF Score': highlighted_avg_tf_scores,
        'Average TF-IDF Score': highlighted_avg_tfidf_scores
    })
    st.session_state['words_to_check'] = [
        {
            'Term': highlighted_feature_names[i],
            'Average TF Score': highlighted_avg_tf_scores[i],
            'Average TF-IDF Score': highlighted_avg_tfidf_scores[i]
        }
        for i in range(len(highlighted_feature_names))
    ]
    st.session_state['analysis_completed'] = True

    # Plot the stacked bar chart using st.bar_chart
    st.subheader('Top 50 Words - Average TF and TF-IDF Scores')
    chart_data = st.session_state['chart_data'].set_index('Terms')
    st.bar_chart(chart_data)

def display_editor():
    # Add a button to start a new analysis
    if st.button('Start a New Analysis'):
        st.session_state.clear()
        st.rerun()

    # Update sidebar label
    st.sidebar.subheader('Optimize Your Content with These Words')
    words_to_check = st.session_state['words_to_check']

    # Create a Quill editor for inputting and editing text
    text_input = st_quill(placeholder='Start typing your content here...', key='quill')

    # Add CSS to adjust the height of the editor
    st.markdown("""
        <style>
        .stQuill {
            height: 400px;
        }
        .stQuill > div:nth-child(2) {
            height: 350px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Ensure text_input is a string
    if text_input is None:
        text_input = ""
    else:
        # Remove HTML tags from the Quill editor output
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text_input, 'html.parser')
        text_input = soup.get_text()

    # Calculate the word count dynamically
    word_count = len(text_input.split()) if text_input.strip() else 0

    # Display the word count and optimization score
    if text_input.strip():
        # Calculate TF scores for the editor content
        total_words = word_count
        tf_vectorizer = CountVectorizer(vocabulary=[word['Term'] for word in words_to_check], ngram_range=(1, 3))
        text_tf_matrix = tf_vectorizer.fit_transform([text_input]).toarray()
        editor_tf_scores = (text_tf_matrix[0] / total_words) * 1000  # Calculate TF and scale

        # Create DataFrame for comparison
        comparison_chart_data = pd.DataFrame({
            'Terms': [word['Term'] for word in words_to_check],
            'Average TF Score': [word['Average TF Score'] for word in words_to_check],
            'Editor TF Score': editor_tf_scores
        })

        # Calculate Occurrences
        occurrences_list = []
        for term in words_to_check:
            occurrences = len(re.findall(r'\b' + re.escape(term['Term']) + r'\b', text_input, flags=re.IGNORECASE))
            occurrences_list.append(occurrences)

        comparison_chart_data['Occurrences'] = occurrences_list

        # Calculate Target, Delta, Min and Max Occurrences
        comparison_chart_data['Target'] = np.maximum(1, np.floor(comparison_chart_data['Average TF Score'] * word_count / 1000).astype(int))
        comparison_chart_data['Delta'] = np.maximum(1, (0.1 * comparison_chart_data['Target']).astype(int))
        comparison_chart_data['Min Occurrences'] = np.maximum(1, comparison_chart_data['Target'] - comparison_chart_data['Delta'])
        comparison_chart_data['Max Occurrences'] = comparison_chart_data['Target'] + comparison_chart_data['Delta']

        # Assign Weights to Terms
        num_terms = len(words_to_check)
        weights = np.ones(num_terms)
        weights[:5] = 6
        weights[5:10] = 2
        comparison_chart_data['Weight'] = weights

        # Compute Term Scores
        term_scores = []
        for idx, row in comparison_chart_data.iterrows():
            occurrences = row['Occurrences']
            min_occurrences = row['Min Occurrences']
            max_occurrences = row['Max Occurrences']
            delta = row['Delta']

            # Calculate deviation and term score
            if occurrences < min_occurrences:
                deviation = (min_occurrences - occurrences) / delta
                deviation = min(1, deviation)
                score = 1 - deviation
            elif occurrences > max_occurrences:
                deviation = (occurrences - max_occurrences) / delta
                deviation = min(1, deviation)
                score = 1 - deviation*2
            else:
                score = 1  # Perfect match
            term_scores.append(score)

        comparison_chart_data['Term Score'] = term_scores
        comparison_chart_data['Weighted Term Score'] = comparison_chart_data['Term Score'] * comparison_chart_data['Weight']

        # Compute Optimization Score
        max_weighted_sum = np.sum(weights)  # Maximum possible score
        optimization_score = (comparison_chart_data['Weighted Term Score'].sum() / max_weighted_sum) * 100

        # Display word count and optimization score
        st.markdown(f"""
            <div style='display: flex; justify-content: space-between; padding: 15px; background-color: #f0f0f0; border-radius: 10px;'>
                <div style='font-size: 18px; color: #004d99;'>
                    Word Count: <strong>{word_count}</strong>
                </div>
                <div style='font-size: 18px; color: #990000;'>
                    Optimization Score: <strong>{optimization_score:.2f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # **Visualization: Bar and Line Chart**
        st.subheader('Comparison of Average TF and Your Content TF Scores')

        # Create the bar and line chart using Altair
        base = alt.Chart(comparison_chart_data).encode(
            x=alt.X('Terms:N', sort='-y', title='Terms')
        )

        bar = base.mark_bar(color='#6495ED').encode(
            y=alt.Y('Average TF Score:Q', title='TF Scores')
        )

        line = base.mark_line(color='#FFAA00', point=alt.OverlayMarkDef(color='#FFAA00')).encode(
            y=alt.Y('Editor TF Score:Q')
        )

        combined_chart = (bar + line).properties(width=700, height=400).interactive()
        st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.markdown(f"Word Count: {word_count}")

    # Display words to check in the sidebar
    with st.sidebar:
        # Update sidebar label
        st.markdown("<div style='text-align: center; font-size: 24px; font-weight: bold;'>Word Frequency</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding: 5px; background-color: #f8f9fa; border-radius: 15px;'>", unsafe_allow_html=True)
        if text_input.strip():
            for idx, row in comparison_chart_data.iterrows():
                word = row['Terms']
                occurrences = row['Occurrences']
                min_occurrences = row['Min Occurrences']
                max_occurrences = row['Max Occurrences']
                target = row['Target']

                # Determine color based on occurrences
                if occurrences < min_occurrences:
                    color = "#E3E3E3"  # Light Blue
                elif occurrences > max_occurrences:
                    color = "#EE2222"  # Dark Blue
                else:
                    color = "#b0DD7c"  # Medium Blue

                # Display term with color and occurrence info
                st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0px; padding: 8px; background-color: {color}; color: black; border-radius: 5px;'>"
                            f"<span style='font-weight: bold;'>{word}</span>"
                            f"<span>{occurrences} / ({min_occurrences}-{max_occurrences})</span>"
                            f"</div>", unsafe_allow_html=True)

                # Calculate progress toward minimum occurrences
                if min_occurrences > 0:
                    progress = min(1.0, occurrences / min_occurrences)
                else:
                    progress = 1.0 if occurrences > 0 else 0

                # Display the progress bar
                st.progress(progress)
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.title('Needs More Words! Optimize Your Content')

    # Initialize session state
    if 'analysis_completed' not in st.session_state:
        st.session_state['analysis_completed'] = False
    if 'editor_started' not in st.session_state:
        st.session_state['editor_started'] = False

    if not st.session_state['editor_started']:
        keyword = st.text_input('Enter a keyword to retrieve content:')
        start_analysis = st.button('Start Analysis')

        if keyword and start_analysis:
            perform_analysis(keyword)

        if st.session_state['analysis_completed'] and not st.session_state['editor_started']:
            continue_to_editor = st.button('Continue to Editor')
            if continue_to_editor:
                st.session_state['editor_started'] = True
                st.rerun()

    if st.session_state.get('editor_started', False):
        display_editor()

if __name__ == '__main__':
    main()
