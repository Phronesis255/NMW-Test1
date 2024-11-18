import re
import math
import time
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD  # Added TruncatedSVD for LSA

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

# Function to lemmatize text
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

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
                title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"

                # Extract favicon
                icon_link = soup.find('link', rel=lambda x: x and ('icon' in x.lower()))
                if icon_link and icon_link.get('href'):
                    favicon_url = urljoin(url, icon_link['href'])
                else:
                    # Default to /favicon.ico
                    favicon_url = urljoin(url, '/favicon.ico')

                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                return title, content.strip(), favicon_url
            else:
                # Return None for content if status code is not 200
                return None, "", ""
        except requests.RequestException:
            # Continue to next attempt
            pass
        time.sleep(2)  # Wait before retrying
    return None, "", ""

# Function to get top unique domain results for a keyword (more than 10 URLs)
@st.cache_data
def get_top_unique_domain_results(keyword, num_results=50, max_domains=50):
    try:
        results = []
        domains = set()
        for url in search(keyword, num_results=num_results):
            domain = urlparse(url).netloc
            if domain not in domains:
                domains.add(domain)
                results.append(url)
            if len(results) >= max_domains:
                break
        return results
    except Exception as e:
        st.error(f"Error during Google search: {e}")
        return []

# Function to filter out value-less terms and custom stopwords
def filter_terms(terms):
    custom_stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "way", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "like", "need"])

    filtered_terms = []
    for term in terms:
        # Exclude terms that contain numbers
        if any(char.isdigit() for char in term):
            continue
        doc = nlp(term)
        # Exclude terms that have undesired POS tags
        if any(token.pos_ in ['AUX', 'PRON', 'DET', 'ADP', 'CCONJ', 'NUM', 'SYM', 'PUNCT'] for token in doc):
            continue
        # Use lemmatization and check for stopwords and custom stopwords
        lemma_tokens = [token.lemma_.lower() for token in doc]
        # Exclude terms if any token is a stopword or in custom stopwords
        if any(token in custom_stopwords or token in nlp.Defaults.stop_words for token in lemma_tokens):
            continue
        lemma = ' '.join(lemma_tokens)
        filtered_terms.append(lemma)
    return filtered_terms

def perform_analysis(keyword):
    st.info('Retrieving top search results...')
    progress_bar0 = st.progress(0)
    top_urls = get_top_unique_domain_results(keyword, num_results=100, max_domains=100)
    if not top_urls:
        st.error('No results found.')
        return

    # Initialize lists to store data
    titles = []
    urls = []
    favicons = []
    retrieved_content = []
    successful_urls = []
    word_counts = []  # New list to store word counts    
    max_contents = 25  # Increased from 10 to 25

    for idx, url in enumerate(top_urls):
        # Remove status messages after a short delay
        status_message = st.empty()
        status_message.text(f"Retrieving content from {url}...")
        progress_bar0.progress((len(retrieved_content) / max_contents))
        title, content, favicon_url = extract_content_from_url(url)
        time.sleep(1)
        status_message.empty()  # Remove the status message
        if title is None:
            title = "No Title"
        if content:
            if len(retrieved_content) < max_contents:
                retrieved_content.append(content)
                successful_urls.append(url)
                titles.append(title)
                favicons.append(favicon_url)
                word_count = len(content.split())
                word_counts.append(word_count)
                
            else:
                # Already have enough content, break the loop
                progress_bar0.empty()
                break
        time.sleep(0.5)  # Shorter delay since messages are removed immediately
        if len(retrieved_content) >= max_contents:
            progress_bar0.empty()
            break

    if len(retrieved_content) < max_contents:
        st.warning(f"Only retrieved {len(retrieved_content)} out of {max_contents} required contents.")

    if not retrieved_content:
        st.error('Failed to retrieve content from the URLs.')
        return

    #calculating ideal word count
    ideal_word_count = int(np.median(word_counts))
    st.session_state['ideal_word_count'] = ideal_word_count  # Store in session state

    documents = retrieved_content

    # Lemmatize the documents
    documents_lemmatized = [lemmatize_text(doc) for doc in documents]
    
    # Display the table of top search results
    st.subheader('Top Search Results')

    # Create a header for the Markdown table
    # Populate the table with search results
    for idx in range(len(titles)):
        favicon_url = favicons[idx]
        title = titles[idx]
        url = successful_urls[idx]
        word_count = word_counts[idx]

        col1, col2 = st.columns([1, 9])
        with col1:
            st.image(favicon_url, width=32)
        with col2:
            st.markdown(
                f"""
                <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: white;">
                    <div style="display: flex; align-items: center;">
                        <img src="{favicon_url}" width="32" style="margin-right: 10px;">
                        <div>
                            <strong>{title}</strong> ({word_count} words)<br>
                            <a href="{url}" target="_blank">{url}</a>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
                        

    if ideal_word_count:
        lower_bound = (ideal_word_count // 500) * 500
        upper_bound = lower_bound + 500
                
        st.info(f"**Suggested Word Count:** Aim for approximately {lower_bound} to {upper_bound} words based on top-performing content.")

    # Continue with the analysis
    # Initialize TF and TF-IDF Vectorizers with n-grams (uni-, bi-, tri-grams)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tf_vectorizer = CountVectorizer(ngram_range=(1, 3))

    # Fit the model and transform the documents into TF and TF-IDF matrices
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents_lemmatized).toarray()
    tf_matrix = tf_vectorizer.fit_transform(documents_lemmatized).toarray()

    # Extract feature names (terms)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Filter feature names to exclude less informative words
    filtered_feature_names = filter_terms(feature_names)

    # Filter TF-IDF and TF matrices to only include filtered terms
    filtered_indices = [i for i, term in enumerate(feature_names) if term in filtered_feature_names]
    tfidf_matrix_filtered = tfidf_matrix[:, filtered_indices]
    tf_matrix_filtered = tf_matrix[:, filtered_indices]

    # Update feature names after filtering
    filtered_feature_names = [feature_names[i] for i in filtered_indices]

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
    st.bar_chart(chart_data, color=["#FFAA00", "#6495ED"])

    # --- Existing LDA Analysis (Adjusted with increased data) ---
    st.subheader('LDA Topic Modeling Results')

    # Create a CountVectorizer for LDA (with the same preprocessing)
    lda_vectorizer = CountVectorizer(ngram_range=(2, 3), vocabulary=filtered_feature_names)
    lda_matrix = lda_vectorizer.fit_transform(documents_lemmatized)

    # Set the number of topics
    num_topics = 5  # You can adjust this number

    # Initialize and fit the LDA model
    lda_model = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=10,
        learning_method='online',
        learning_decay=0.7,
        random_state=42
    )
    lda_model.fit(lda_matrix)

    # Get the topics and their top terms
    lda_feature_names = lda_vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-15:][::-1]  # Get indices of top 10 terms
        top_terms = [lda_feature_names[i] for i in top_indices]
        topics[f'Topic {topic_idx + 1}'] = top_terms

    # Display the topics in a table
    topics_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in topics.items()]))
    st.table(topics_df)

    # Visualize the topics using a bar chart
    st.subheader('LDA Topics Term Distribution')

    # Prepare data for visualization
    topic_term_data = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_terms = [lda_feature_names[i] for i in top_indices]
        top_weights = topic[top_indices]
        for term, weight in zip(top_terms, top_weights):
            topic_term_data.append({
                'Topic': f'Topic {topic_idx + 1}',
                'Term': term,
                'Weight': weight
            })

    topic_term_df = pd.DataFrame(topic_term_data)

    # Create a bar chart for each topic
    for topic_idx in range(num_topics):
        topic_label = f'Topic {topic_idx + 1}'
        topic_data = topic_term_df[topic_term_df['Topic'] == topic_label]
        chart = alt.Chart(topic_data).mark_bar().encode(
            x=alt.X('Weight:Q', title='Term Weight'),
            y=alt.Y('Term:N', sort='-x'),
            tooltip=['Term', 'Weight']
        ).properties(
            title=topic_label,
            width=600,
            height=300
        )
        st.altair_chart(chart)

    # --- New Section: LSA Analysis ---
    st.subheader('LSA Topic Modeling Results')

    # Perform LSA using TruncatedSVD
    num_topics_lsa = 5  # You can adjust this number
    lsa = TruncatedSVD(n_components=num_topics_lsa, n_iter=100, random_state=42)
    lsa_matrix = lsa.fit_transform(tfidf_matrix_filtered)

    # Get the topics and their top terms
    lsa_components = lsa.components_
    lsa_feature_names = [filtered_feature_names[i] for i in range(len(filtered_feature_names))]
    lsa_topics = {}
    for topic_idx, topic in enumerate(lsa_components):
        top_indices = topic.argsort()[-15:][::-1]
        top_terms = [lsa_feature_names[i] for i in top_indices]
        lsa_topics[f'Topic {topic_idx + 1}'] = top_terms

    # Display the topics in a table
    lsa_topics_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in lsa_topics.items()]))
    st.table(lsa_topics_df)

    # Visualize the topics using a bar chart
    st.subheader('LSA Topics Term Distribution')

    # Prepare data for visualization
    lsa_topic_term_data = []
    for topic_idx, topic in enumerate(lsa_components):
        top_indices = topic.argsort()[-10:][::-1]
        top_terms = [lsa_feature_names[i] for i in top_indices]
        top_weights = topic[top_indices]
        for term, weight in zip(top_terms, top_weights):
            lsa_topic_term_data.append({
                'Topic': f'Topic {topic_idx + 1}',
                'Term': term,
                'Weight': weight
            })

    lsa_topic_term_df = pd.DataFrame(lsa_topic_term_data)

    # Create a bar chart for each topic
    for topic_idx in range(num_topics_lsa):
        topic_label = f'Topic {topic_idx + 1}'
        topic_data = lsa_topic_term_df[lsa_topic_term_df['Topic'] == topic_label]
        chart = alt.Chart(topic_data).mark_bar().encode(
            x=alt.X('Weight:Q', title='Term Weight'),
            y=alt.Y('Term:N', sort='-x'),
            tooltip=['Term', 'Weight']
        ).properties(
            title=topic_label,
            width=600,
            height=300
        )
        st.altair_chart(chart)

def display_editor():
    # Add a button to start a new analysis
    if st.button('Start a New Analysis'):
        st.session_state.clear()
        st.rerun()

    # Retrieve the ideal word count from session state
    ideal_word_count = st.session_state.get('ideal_word_count', None)

    # Display the ideal word count suggestion
    if ideal_word_count:
        lower_bound = (ideal_word_count // 500) * 500
        upper_bound = lower_bound + 500
                
        st.info(f"**Suggested Word Count:** Aim for approximately {lower_bound} to {upper_bound} words based on top-performing content.")
    
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
        st.markdown("<div style='text-align: center; font-size: 24px; color: #ffaa00;'>Word Frequency</div>", unsafe_allow_html=True)
        st.markdown("<div style='padding: 1px; background-color: #f8f9fa; border-radius: 15px;'>", unsafe_allow_html=True)
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

        if st.session_state.get('analysis_completed', False) and not st.session_state['editor_started']:
            continue_to_editor = st.button('Continue to Editor')
            if continue_to_editor:
                st.session_state['editor_started'] = True
                st.rerun()

    if st.session_state.get('editor_started', False):
        display_editor()

if __name__ == '__main__':
    main()





