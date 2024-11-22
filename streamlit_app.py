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

st.set_page_config(page_title="Needs More Words! Optimize Your Content", page_icon="🔠")

st.markdown("""
    <style>n
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

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

# Function to lemmatize text with custom overrides
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        # Context-aware overrides for specific terms
        if token.text.lower() == "media" and token.lemma_.lower() == "medium":
            lemmatized_tokens.append("media")
        elif token.text.lower() == "data" and token.lemma_.lower() == "datum":
            lemmatized_tokens.append("data")
        elif token.text.lower() == "publishers" and token.lemma_.lower() == "publisher":
            lemmatized_tokens.append("publisher")
        else:
            lemmatized_tokens.append(token.lemma_)
    return ' '.join(lemmatized_tokens)

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

def compute_embedding(text, embeddings_index):
    words = text.lower().split()
    embeddings = []
    for word in words:
        if word in embeddings_index:
            embeddings.append(embeddings_index[word])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return None


@st.cache_resource
def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]  # The word
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

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
    word_counts = []
    max_contents = 5

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
            word_count = len(content.split())            
            if len(retrieved_content) < max_contents:
                retrieved_content.append(content)
                successful_urls.append(url)
                titles.append(title)
                favicons.append(favicon_url)
                if word_count > 1000:
                    word_counts.append(word_count)
                else:
                    word_counts.append(1000)
            else:
                # Already have enough content, break the loop
                progress_bar0.empty()
                break
        time.sleep(0.5)
        if len(retrieved_content) >= max_contents:
            progress_bar0.empty()
            break

    if len(retrieved_content) < max_contents:
        st.warning(f"Only retrieved {len(retrieved_content)} out of {max_contents} required contents.")

    if not retrieved_content:
        st.error('Failed to retrieve sufficient content from the URLs.')
        return

    # Calculating ideal word count
    if word_counts:
        ideal_word_count = int(np.median(word_counts)) + 500
    else:
        ideal_word_count = 1000  # Default value if no word counts
    st.session_state['ideal_word_count'] = ideal_word_count  # Store in session state

    documents = retrieved_content

    # Lemmatize the documents
    documents_lemmatized = [lemmatize_text(doc) for doc in documents]

    # Display the table of top search results
    st.subheader('Top Search Results')

    # Populate the table with search results
    for idx in range(len(titles)):
        favicon_url = favicons[idx]
        title = titles[idx]
        url = successful_urls[idx]
        word_count = word_counts[idx]

        col1, col2 = st.columns([1, 9])
        with col1:
            st.markdown(idx+1)
        with col2:
            st.markdown(
                f"""
                <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: black;">
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
        top_indices = topic.argsort()[-10:][::-1]  # Get indices of top 10 terms
        top_terms = [lda_feature_names[i] for i in top_indices]
        topics[f'Topic {topic_idx + 1}'] = top_terms

    # Display the topics in a table
    topics_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in topics.items()]))
    st.table(topics_df)

    # Visualize the topics using a bar chart
    st.subheader('LDA Topics Term Distribution')

    # Prepare data for visualization
    lda_top_terms = []    
    topic_term_data = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_terms = [lda_feature_names[i] for i in top_indices]
        top_weights = topic[top_indices]
        lda_top_terms.extend(top_terms)

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
    # Combine LDA terms with TF-IDF terms
    combined_terms_set = set(lda_top_terms + filtered_feature_names)
    combined_terms = list(combined_terms_set)




    # --- New Section: Combining TF-IDF and GloVe Embeddings ---

    # Load GloVe embeddings
    embeddings_index = load_glove_embeddings('glove.6B.100d.txt')  # Ensure this file is available

    # Get embeddings for combined terms
    term_embeddings = []
    valid_terms = []
    term_avg_tfidf_scores = []
    term_avg_tf_scores = []
    for term in combined_terms:
        # Use the first word of the term for simplicity, or average embeddings for multi-word terms
        words = term.split()
        embeddings = []
        for word in words:
            if word in embeddings_index:
                embeddings.append(embeddings_index[word])
        if embeddings:
            # Average embeddings for multi-word terms
            term_embedding = np.mean(embeddings, axis=0)
            term_embeddings.append(term_embedding)
            valid_terms.append(term)
            # Retrieve TF-IDF and TF scores if available
            if term in filtered_feature_names:
                idx = filtered_feature_names.index(term)
                term_avg_tfidf_scores.append(avg_tfidf_scores_scaled[idx])
                term_avg_tf_scores.append(avg_tf_scores[idx])
            else:
                # Assign default scores for terms from LDA only
                term_avg_tfidf_scores.append(np.mean(avg_tfidf_scores_scaled) * 0.5)
                term_avg_tf_scores.append(np.mean(avg_tf_scores) * 0.5)
    if not term_embeddings:
        st.warning('No embeddings found for the terms.')
        return

    term_embeddings = np.array(term_embeddings)

    # Compute the embedding for the keyword
    keyword_embedding = compute_embedding(keyword, embeddings_index)
    if keyword_embedding is None:
        st.warning('No embedding found for the keyword.')
        return

    # Compute cosine similarity between the keyword and each term
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity([keyword_embedding], term_embeddings)[0]  # shape: (num_terms,)

    # Combine TF-IDF scores and similarities
    combined_scores = np.array(term_avg_tfidf_scores) * similarities  # Element-wise multiplication

    # Get top N terms based on combined scores
    N = 50
    top_indices = np.argsort(combined_scores)[-N:][::-1]
    top_terms = [valid_terms[i] for i in top_indices]
    top_scores = [combined_scores[i] for i in top_indices]
    top_avg_tfidf_scores = [term_avg_tfidf_scores[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]
    top_avg_tf_scores = [term_avg_tf_scores[i] for i in top_indices]

    # Store the results in session state
    st.session_state['chart_data'] = pd.DataFrame({
        'Terms': top_terms,
        'Combined Score': top_scores,
        'Average TF-IDF Score': top_avg_tfidf_scores,
        'Similarity to Keyword': top_similarities
    })
    st.session_state['words_to_check'] = [
        {
            'Term': top_terms[i],
            'Average TF Score': top_avg_tf_scores[i],
            'Average TF-IDF Score': top_avg_tfidf_scores[i]
        }
        for i in range(len(top_terms))
    ]
    st.session_state['analysis_completed'] = True


    # Plot the stacked bar chart using st.bar_chart
    st.subheader('Top 50 Words - Average TF and TF-IDF Scores')
    chart_data = st.session_state['chart_data'].set_index('Terms')
    st.bar_chart(chart_data, color=["#FFAA00", "#6495ED","#FF5511"])

    # Plot the bar chart
    st.subheader('Top Relevant Words - Combined Scores')
    chart_data = st.session_state['chart_data'].set_index('Terms')
    st.bar_chart(chart_data['Combined Score'])


    # Display the table of top terms with scores
    st.table(st.session_state['chart_data'])


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

    # Remove HTML tags from the Quill editor output
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text_input, 'html.parser')
    text_input = soup.get_text()

    # Lemmatize the user's input text
    text_input_lemmatized = lemmatize_text(text_input)

    # Calculate the word count dynamically
    word_count = len(text_input.split()) if text_input.strip() else 0

    # --- New Section: Import SEO Keywords ---
    # Add a button to import CSV files
    st.subheader('Import SEO Keywords')
    uploaded_file = st.file_uploader('Upload a CSV file from your SEO keyword research tool:', type='csv')

    # Initialize imported_keywords
    imported_keywords = None

    if uploaded_file is not None:
        # Read the CSV file
        df_seo = pd.read_csv(uploaded_file)

        # Remove unnecessary columns
        df_seo = df_seo[['Keyword', 'Avg. Search Volume (Last 6 months)', 'Keyword Difficulty']]

        # Convert columns to numeric, handling errors
        df_seo['Avg. Search Volume (Last 6 months)'] = pd.to_numeric(df_seo['Avg. Search Volume (Last 6 months)'], errors='coerce')
        df_seo['Keyword Difficulty'] = pd.to_numeric(df_seo['Keyword Difficulty'], errors='coerce')

        # Calculate a score to rank keywords (e.g., search volume divided by difficulty)
        df_seo['Score'] = df_seo['Avg. Search Volume (Last 6 months)'] / (df_seo['Keyword Difficulty'] + 1e-6)

        # Sort the keywords by the score in descending order
        df_seo = df_seo.sort_values(by='Score', ascending=False)

        # Keep only the top 5 keywords
        df_seo = df_seo.head(5).reset_index(drop=True)

        # Store the imported keywords in session state
        st.session_state['imported_keywords'] = df_seo

        imported_keywords = df_seo
    else:
        # Retrieve from session state
        imported_keywords = st.session_state.get('imported_keywords', None)

    # Get words_to_check (TF-IDF terms)
    words_to_check = st.session_state['words_to_check']

    # Create a set of TF-IDF terms
    tfidf_terms_set = set(word['Term'] for word in words_to_check)

    # Prepare the list of imported keywords that are not duplicates
    imported_words_to_check = []
    if imported_keywords is not None and not imported_keywords.empty:
        max_search_volume = imported_keywords['Avg. Search Volume (Last 6 months)'].max()
        for idx, row in imported_keywords.iterrows():
            term = row['Keyword']
            if term in tfidf_terms_set:
                # If duplicate, ignore the imported keyword and display it as a TF-IDF term
                continue  # Skip adding this imported keyword
            else:
                # Add the imported keyword
                search_volume = row['Avg. Search Volume (Last 6 months)']
                difficulty = row['Keyword Difficulty']
                # Normalize search volume for weighting
                weight = 3  # Base weight for imported keywords
                if pd.notna(search_volume) and max_search_volume > 0:
                    weight += (search_volume / max_search_volume) * 2  # Scale weight between 3 and 5
                else:
                    weight += 1  # Default weight if search volume is not available
                imported_words_to_check.append({
                    'Term': term,
                    'Average TF Score': 0,
                    'Average TF-IDF Score': 0,
                    'Weight': weight,
                    'Search Volume': search_volume,
                    'Keyword Difficulty': difficulty,
                    'IsImported': True
                })
    else:
        imported_keywords = pd.DataFrame()  # Empty DataFrame

    # Now, combine the lists, with imported keywords at the top
    combined_words_to_check = imported_words_to_check + words_to_check  # words_to_check already includes duplicates

    # Display the word count and optimization score
    if text_input.strip():
        # Calculate TF scores for the editor content
        total_words = word_count
        tf_vectorizer = CountVectorizer(vocabulary=[word['Term'] for word in combined_words_to_check], ngram_range=(1, 3))
        text_tf_matrix = tf_vectorizer.transform([text_input_lemmatized]).toarray()
        editor_tf_scores = (text_tf_matrix[0] / total_words) * 1000 if total_words > 0 else np.zeros(len(combined_words_to_check))  # Calculate TF and scale

        # Retrieve average TF scores from analysis
        average_tf_scores = np.array([word.get('Average TF Score', 0) for word in combined_words_to_check])

        # Assign Weights to Terms (already assigned during term preparation)
        weights = np.array([word.get('Weight', 1) for word in combined_words_to_check])

        # Create DataFrame for comparison
        comparison_chart_data = pd.DataFrame({
            'Terms': [word['Term'] for word in combined_words_to_check],
            'Average TF Score': [word.get('Average TF Score', 0) for word in combined_words_to_check],
            'Editor TF Score': editor_tf_scores,
            'IsImported': [word.get('IsImported', False) for word in combined_words_to_check],
            'Weight': [word.get('Weight', 1) for word in combined_words_to_check]
        })

        # Calculate Occurrences
        occurrences_list = []
        for term in combined_words_to_check:
            lemmatized_term = lemmatize_text(term['Term'])            
            occurrences = len(re.findall(r'\b' + re.escape(lemmatized_term) + r'\b', text_input_lemmatized, flags=re.IGNORECASE))
            occurrences_list.append(occurrences)
        comparison_chart_data['Occurrences'] = occurrences_list

        # Calculate Target, Delta, Min and Max Occurrences
        targets = []
        deltas = []
        for idx, word in comparison_chart_data.iterrows():
            if word['IsImported']:
                # Calculate target occurrences based on word count
                target = max(1, int(word_count / 500))
                delta = max(1, int(0.1 * target))
            else:
                # Existing calculation for TF-IDF terms
                target = np.maximum(1, int(np.floor(word['Average TF Score'] * word_count / 1000)))
                delta = np.maximum(1, int(0.1 * target))
            targets.append(target)
            deltas.append(delta)
        comparison_chart_data['Target'] = targets
        comparison_chart_data['Delta'] = deltas
        comparison_chart_data['Min Occurrences'] = np.maximum(1, comparison_chart_data['Target'] - comparison_chart_data['Delta'])
        comparison_chart_data['Max Occurrences'] = comparison_chart_data['Target'] + comparison_chart_data['Delta']

        # Compute Term Scores
        term_scores = []
        for editor_tf_score, average_tf_score in zip(editor_tf_scores, average_tf_scores):
            if average_tf_score > 0:
                deviation = abs(editor_tf_score - average_tf_score) / average_tf_score
                term_score = max(0, 1 - deviation)
            else:
                # If the average TF score is zero
                if editor_tf_score == 0:
                    term_score = 1  # Perfect match (both are zero)
                else:
                    term_score = 0  # Editor uses the term, but it's not used in top content
            term_scores.append(term_score)

        term_scores = np.array(term_scores)
        weighted_term_scores = term_scores * weights

        # Compute Optimization Score
        max_weighted_sum = np.sum(weights)  # Maximum possible score
        optimization_score = (np.sum(weighted_term_scores) / max_weighted_sum) * 100 if max_weighted_sum > 0 else 0

        # Ensure Optimization Score does not drop below 2%
        optimization_score = max(optimization_score, 2)

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
                    color = "#E3E3E3"  # Light Gray
                elif occurrences > max_occurrences:
                    color = "#EE2222"  # Red
                else:
                    color = "#b0DD7c"  # Green

                # Check if the term is an imported keyword
                if row['IsImported']:
                    # Different styling for imported keywords
                    background_style = 'background-color: #E6FFE6;'  # Light green
                else:
                    background_style = f'background-color: {color};'

                st.markdown(f"<div style='display: flex; flex-direction: column; margin-bottom: 5px; padding: 8px; {background_style} color: black; border-radius: 5px;'>"
                            f"<span style='font-weight: bold;'>{word}</span>"
                            f"<span>Occurrences: {occurrences} / Target: ({min_occurrences}-{max_occurrences})</span>"
                            f"</div>", unsafe_allow_html=True)

                # Calculate progress toward minimum occurrences
                if min_occurrences > 0:
                    progress = min(1.0, occurrences / min_occurrences)
                else:
                    progress = 1.0 if occurrences > 0 else 0

                # Display the progress bar
                st.progress(progress)
        st.markdown("</div>", unsafe_allow_html=True)

    # Display imported keywords with their search volume and difficulty at the bottom
    with st.sidebar:
        if imported_keywords is not None and not imported_keywords.empty:
            st.markdown("<div style='text-align: center; font-size: 20px; color: #2E8B57;'>Imported SEO Keywords</div>", unsafe_allow_html=True)
            for idx, row in imported_keywords.iterrows():
                keyword = row['Keyword']
                search_volume = row['Avg. Search Volume (Last 6 months)']
                difficulty = row['Keyword Difficulty']
                occurrences = text_input.lower().count(keyword.lower())
                st.markdown(f"<div style='padding: 8px; background-color: #E6FFE6; color: black; border-radius: 5px; margin-bottom: 5px;'>"
                            f"<strong>{keyword}</strong><br>"
                            f"Occurrences: {occurrences}<br>"
                            f"Avg. Search Volume (6 months): {search_volume}<br>"
                            f"Keyword Difficulty: {difficulty}"
                            f"</div>", unsafe_allow_html=True)




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
