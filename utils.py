# utils.py
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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
from streamlit_quill import st_quill
from googlesearch import search
import altair as alt

import people_also_ask as paa
from transformers import pipeline
from transformers.pipelines import QuestionAnsweringPipeline
from typing import List, Dict
import base64

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import nltk
import string
import ssl
import torch

# Initialize NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        from spacy.cli import download
        download('en_core_web_sm')
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

def get_sentence_embedding(sentence, embeddings_index):
    words = sentence.lower().split()
    embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if embeddings:
        sentence_embedding = np.mean(embeddings, axis=0)
    else:
        sentence_embedding = np.zeros(100)  # Assuming 100-dimensional embeddings
    return sentence_embedding

# Define all your shared functions here
def lemmatize_text(text):
    # Your lemmatization code
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



#Cache sentence transformer model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def remove_duplicate_questions(questions, similarity_threshold=0.75):
    # Preprocess questions
    def preprocess(text):
        # Lowercase, remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    preprocessed_questions = [preprocess(q) for q in questions]

    # Encode questions using SentenceTransformer
    model = load_embedding_model()
    embeddings = model.encode(preprocessed_questions)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Cluster questions
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        linkage='complete',
        distance_threshold=1 - similarity_threshold
    )
    clustering_model.fit(1 - similarity_matrix)  # Convert similarity to distance

    cluster_labels = clustering_model.labels_

    # Select a representative question from each cluster
    cluster_to_questions = {}
    for idx, label in enumerate(cluster_labels):
        if label not in cluster_to_questions:
            cluster_to_questions[label] = [questions[idx]]
        else:
            cluster_to_questions[label].append(questions[idx])

    # For each cluster, select the shortest question as representative
    representative_questions = []
    for cluster_questions in cluster_to_questions.values():
        representative = min(cluster_questions, key=len)
        representative_questions.append(representative)

    return representative_questions


# Function to extract content from a URL with retries and user-agent header
@st.cache_data
def extract_content_from_url(url, extract_headings=False, retries=2, timeout=5):
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
                headings = []                
                if extract_headings:
                    for level in ['h2', 'h3', 'h4']:
                        for tag in soup.find_all(level):
                            headings.append({'level': level, 'text': tag.get_text(strip=True)})
                # Extract favicon
                icon_link = soup.find('link', rel=lambda x: x and ('icon' in x.lower()))
                if icon_link and icon_link.get('href'):
                    favicon_url = urljoin(url, icon_link['href'])
                else:
                    # Default to /favicon.ico
                    favicon_url = urljoin(url, '/favicon.ico')

                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                return title, content.strip(), favicon_url, headings
            else:
                # Return None for content if status code is not 200
                return None, "", "", None
        except requests.RequestException:
            # Continue to next attempt
            pass
        time.sleep(2)  # Wait before retrying
    return None, "", "", None

#Function to create qa pipeline with transformers
@st.cache_resource
def load_qa_pipeline():
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return qa_pipeline

#chunks text
def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    sentences = text.split('.')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_size:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '.'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def is_question_answered(qa_pipeline: QuestionAnsweringPipeline, question: str, context: str, threshold: float = 0.5) -> Dict:
    try:
        result = qa_pipeline(question=question, context=context)
        if result['score'] >= threshold:
            return {'answered': True, 'answer': result['answer'], 'score': result['score']}
        else:
            return {'answered': False, 'answer': None, 'score': result['score']}
    except Exception as e:
        return {'answered': False, 'answer': None, 'score': 0.0}


# Function to get top unique domain results for a keyword (more than 10 URLs)
@st.cache_data
def get_top_unique_domain_results(keyword, num_results=50, max_domains=50):
    try:
        results = []
        domains = set()
        for url in search(keyword, num_results=num_results, lang="en"):
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

import difflib
from urllib.parse import urlparse
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_brand_name(url, title):
    # Extract the domain name
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    
    # Remove common prefixes (e.g., 'www')
    if domain_parts[0] == 'www':
        domain_parts.pop(0)

    # Take the root domain (e.g., 'lawruler' from 'lawruler.com')
    domain_root = domain_parts[0].capitalize()

    # Attempt to extract the brand name from the title
    if title:
        # Split title into parts (e.g., "Client Intake Software - Law Ruler")
        title_parts = title.split(' - ')
        for part in reversed(title_parts):  # Start from the end
            ratio = difflib.SequenceMatcher(None, domain_root.lower(), part.lower()).ratio()
            if ratio > 0.8:  # Threshold for fuzzy matching
                return part.strip()  # Return the part containing the domain root

    # Fallback to the root domain as the brand name
    return domain_root

def is_brand_mentioned(term, brand_name):
    # Case-insensitive check
    if brand_name.lower() in term.lower():
        return True

    # Optionally use difflib for partial matches
    ratio = difflib.SequenceMatcher(None, term.lower().replace(' ', ''), brand_name.lower().replace(' ', '')).ratio()
    # Consider it a match if similarity is high enough (adjust threshold as needed)
    if ratio > 0.8:
        return True

    # Check NER: if a named entity matches or closely resembles the brand
    doc = nlp(term)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'PERSON', 'GPE']:
            # Compare entity text with brand_name using a ratio
            ratio_ent = difflib.SequenceMatcher(None, ent.text.lower().replace(' ', ''), brand_name.lower().replace(' ', '')).ratio()
            if ratio_ent > 0.8:
                return True

    return False


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


def is_not_branded(question):
    """
    Checks if a given question string is branded.

    Args:
        question (str): The question string to check.

    Returns:
        bool: True if the question is not branded, False otherwise.
    """
    # Retrieve brand names from session state
    brands = st.session_state.get('brands', [])
    
    # Check if the question mentions any brand name
    for brand in brands:
        if is_brand_mentioned(question, brand):
            return False  # The question is branded
    return True  # The question is not branded


def perform_analysis(keyword):
    start_time = time.time()
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
    max_contents = 12
    headings_data = []
    brand_names = set()  # To store unique brand names
    
    for idx, url in enumerate(top_urls):
        # Remove status messages after a short delay
        status_message = st.empty()
        status_message.text(f"Retrieving content from {url}...")
        progress_bar0.progress((len(retrieved_content) / max_contents))
        title, content, favicon_url, headings = extract_content_from_url(url, extract_headings=True)
        time.sleep(1)
        status_message.empty()  # Remove the status message
        if title is None:
            title = "No Title"

        brand_name = extract_brand_name(url, title)
        brand_names.add(brand_name)


        if headings and isinstance(headings, list):  # Ensure headings is a list
            for heading in headings:
                if isinstance(heading, dict) and 'text' in heading:  # Ensure heading is a dictionary with 'text'
                    # Append title along with heading text and URL
                    headings_data.append({
                        'text': heading['text'].strip(),
                        'url': url,
                        'title': title
                    })
                else:
                    st.warning(f"Unexpected heading format: {heading}")


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

    # Save unique brand names to session state
    st.session_state['brands'] = list(brand_names)

    # Calculating ideal word count
    if word_counts:
        ideal_word_count = int(np.median(word_counts)) + 500
    else:
        ideal_word_count = 1000  # Default value if no word counts
    st.session_state['ideal_word_count'] = ideal_word_count  # Store in session state

    documents = retrieved_content

    # Lemmatize the documents
    documents_lemmatized = [lemmatize_text(doc) for doc in documents]
    
    
    if headings_data:
        st.subheader("All Headings in the Content")
        # Create a DataFrame for headings
        question_words = ['how', 'why', 'what', 'who', 'which', 'is', 'are', 'can', 'does', 'will']
        filtered_headings_data = [
            heading for heading in headings_data
            if (heading['text'].endswith('?') or
                (heading['text'].split() and heading['text'].split()[0].lower() in question_words))
        ]

        # Remove duplicates based on both text and URL
        filtered_headings_data = list({(heading['text'], heading['url'], heading.get('title', 'No Title')) for heading in filtered_headings_data})

        # Convert to DataFrame
        paa_list_df = pd.DataFrame(filtered_headings_data, columns=['Question', 'URL', 'Title'])

        # Update session state
        st.session_state['paa_list'] = paa_list_df
        
        # Ensure "Keep" key is added to paa_list
        # if 'paa_list' in st.session_state:
        #     paa_list = st.session_state['paa_list']
        #     if isinstance(paa_list, list) and all(isinstance(item, str) for item in paa_list):
        #         st.session_state['paa_list'] = [{"Question": q, "Keep": True} for q in paa_list]
        #     elif isinstance(paa_list, list) and all(isinstance(item, dict) for item in paa_list):
        #         paa_df = pd.DataFrame(paa_list)
        #         if 'Keep' not in paa_df.columns:
        #             paa_df['Keep'] = True
        #         paa_list = paa_df.to_dict(orient='records')
        #         st.session_state['paa_list'] = paa_df

        # Display the cleaned-up questions
        st.write("Total Questions Extracted and Cleaned:", len(paa_list_df))
        # st.table(paa_list_df)

        # Create a DataFrame for headings
        headings_df = pd.DataFrame(filtered_headings_data)
        st.session_state['headings_df'] = headings_df
        # Display the DataFrame using Streamlit
        headings_csv = headings_df.to_csv(index=False)
        b64 = base64.b64encode(headings_csv.encode()).decode()
        st.markdown(f'<a href="data:file/headings_csv;base64,{b64}" download="results.csv">Download Results</a>', unsafe_allow_html=True)

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
    lda_vectorizer = CountVectorizer(ngram_range=(2, 4), vocabulary=filtered_feature_names)
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
    lda_top_terms = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]  # Get indices of top 10 terms
        top_terms = [lda_feature_names[i] for i in top_indices]
        topics[f'Topic {topic_idx + 1}'] = top_terms
        lda_top_terms.extend(top_terms)


    # Display the topics in a table
    topics_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in topics.items()]))
    st.table(topics_df)

    lda_top_terms = list(set(lda_top_terms))
    non_branded_lda_terms = [term for term in lda_top_terms if is_not_branded(term)]  # Apply is_not_branded
    lda_top_terms = non_branded_lda_terms
    if len(lda_top_terms) > 50:
        lda_top_terms = lda_top_terms[:50]
    
    st.session_state['lda_top_terms'] = lda_top_terms
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

    similarities = cosine_similarity([keyword_embedding], term_embeddings)[0]  # shape: (num_terms,)

    # Combine TF-IDF scores and similarities
    combined_scores = np.array(term_avg_tfidf_scores) * similarities  # Element-wise multiplication

    # Get top N terms based on combined scores
    N = 50
    top_indices = np.argsort(combined_scores)[-N:][::-1]
    top_terms = [valid_terms[i] for i in top_indices]
    top_scores = [combined_scores[i] for i in top_indices]
    top_avg_tfidf_scores = [term_avg_tfidf_scores[i] for i in top_indices]
    top_similarities = [similarities[i]*100 for i in top_indices]
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
    elapsed_time = time.time() - start_time
    print(f"Time taken to reach Content Editor: {elapsed_time:.2f} seconds")        

    st.session_state['analysis_completed'] = True


    # Plot the stacked bar chart using st.bar_chart
    # # Plot the bar chart
    # st.subheader('Top Relevant Words - Combined Scores')
    # chart_data = st.session_state['chart_data'].set_index('Terms')
    # st.bar_chart(chart_data['Combined Score'])


    # # Display the table of top terms with scores
    # st.table(st.session_state['chart_data'])


def display_editor():
    # Add a button to start a new analysis
    if st.button('Start a New Analysis'):
        st.session_state.clear()
        st.rerun()

    # # Add a unique key to the 'Start a New Analysis' button
    # if st.button('Edit List of Important Terms', key='edit_terms_button'):
    #     st.session_state['step'] = 'term_editor'
    #     st.rerun()


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
    text_input_plain = soup.get_text()

    # Lemmatize the user's input text
    text_input_lemmatized = lemmatize_text(text_input_plain)

    # Calculate the word count dynamically
    word_count = len(text_input_plain.split()) if text_input_plain.strip() else 0

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
    # Retrieve average TF scores from analysis
    average_tf_scores = np.array([word.get('Average TF Score', 0) for word in combined_words_to_check])

    # Assign Weights to Terms (already assigned during term preparation)
    weights = np.array([word.get('Weight', 1) for word in combined_words_to_check])
    

    if text_input_plain.strip():
        # Retrieve or create comparison_chart_data
        if 'comparison_chart_data' in st.session_state:
            comparison_chart_data = st.session_state['comparison_chart_data'].copy()
        else:
            # Build comparison_chart_data as before
            comparison_chart_data = pd.DataFrame({
                'Terms': [word['Term'] for word in combined_words_to_check],
                'Average TF Score': [word.get('Average TF Score', 0) for word in combined_words_to_check],
                'IsImported': [word.get('IsImported', False) for word in combined_words_to_check],
                'Weight': [word.get('Weight', 1) for word in combined_words_to_check],
                'Keep': True  # Initialize 'Keep' column
            })

        # Update 'Editor TF Score'
        total_words = word_count
        tf_vectorizer = CountVectorizer(vocabulary=comparison_chart_data['Terms'].tolist(), ngram_range=(1, 3))
        text_tf_matrix = tf_vectorizer.transform([text_input_lemmatized]).toarray()
        editor_tf_scores = (text_tf_matrix[0] / total_words) * 1000 if total_words > 0 else np.zeros(len(comparison_chart_data))
        comparison_chart_data['Editor TF Score'] = editor_tf_scores

        # Update 'Occurrences'
        occurrences_list = []
        for term in comparison_chart_data['Terms']:
            lemmatized_term = lemmatize_text(term)
            occurrences = len(re.findall(r'\b' + re.escape(lemmatized_term) + r'\b', text_input_lemmatized, flags=re.IGNORECASE))
            occurrences_list.append(occurrences)
        comparison_chart_data['Occurrences'] = occurrences_list

        # Calculate Target, Delta, Min and Max Occurrences
        targets = []
        deltas = []
        for idx, row in comparison_chart_data.iterrows():
            if row['IsImported']:
                target = max(1, int(word_count / 500))
                delta = max(1, int(0.1 * target))
            else:
                target = max(1, int(np.floor(row['Average TF Score'] * word_count / 1000)))
                delta = max(1, int(0.1 * target))
            targets.append(target)
            deltas.append(delta)
        comparison_chart_data['Target'] = targets
        comparison_chart_data['Delta'] = deltas
        comparison_chart_data['Min Occurrences'] = np.maximum(1, comparison_chart_data['Target'] - comparison_chart_data['Delta'])
        comparison_chart_data['Max Occurrences'] = comparison_chart_data['Target'] + comparison_chart_data['Delta']

        # Filter based on 'Keep' column
        filtered_chart_data = comparison_chart_data[comparison_chart_data['Keep'] == True].reset_index(drop=True)

        # Compute Optimization Score
        def compute_term_score(occurrences, target, min_occurrences, max_occurrences):
            if min_occurrences <= occurrences <= max_occurrences:
                range_score = 1
            else:
                range_score = 0
            proximity_score = max(0, 1 - abs(occurrences - target) / target)
            term_score = 0.5 * range_score + 0.5 * proximity_score
            return term_score

        def compute_optimization_score(filtered_chart_data):
            term_scores = []
            for idx, row in filtered_chart_data.iterrows():
                occurrences = row['Occurrences']
                target = row['Target']
                min_occurrences = row['Min Occurrences']
                max_occurrences = row['Max Occurrences']
                weight = row['Weight']
                term_score = compute_term_score(occurrences, target, min_occurrences, max_occurrences)
                term_scores.append(term_score)
            term_scores = np.array(term_scores)
            weights = filtered_chart_data['Weight'].values
            weighted_term_scores = term_scores * weights
            max_weighted_sum = np.sum(weights)
            optimization_score = (np.sum(weighted_term_scores) / max_weighted_sum) * 100 if max_weighted_sum > 0 else 0
            optimization_score = max(optimization_score, 19) + 10
            optimization_score = min(optimization_score, 94)
            return optimization_score

        optimization_score = compute_optimization_score(filtered_chart_data)

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

        # Save updated DataFrame back to session state
        st.session_state['comparison_chart_data'] = comparison_chart_data.copy()

        original_keyword = st.session_state.get('keyword', '')
        embeddings_index = load_glove_embeddings('glove.6B.100d.txt')
        original_keyword_embedding = get_sentence_embedding(original_keyword, embeddings_index)
        similarities = []

        
        if 'paa_list' in st.session_state:
            # st.table(st.session_state['paa_list'])
            paa_list = st.session_state['paa_list']
            if isinstance(paa_list, pd.DataFrame):
                # Filter branded questions

                # Apply the branded filter
                non_branded_paa_list = paa_list[paa_list['Question'].apply(is_not_branded)]

                # Calculate similarities for non-branded questions
                for index, row in non_branded_paa_list.iterrows():
                    question = row['Question']
                    question_embedding = get_sentence_embedding(question, embeddings_index)
                    sim = cosine_similarity(
                        original_keyword_embedding.reshape(1, -1),
                        question_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

                # Add similarities to the DataFrame
                non_branded_paa_list['Similarity'] = similarities
                

                # Set a similarity threshold
                similarity_threshold = 0.87  # Adjust based on your requirements

                # Filter questions based on similarity threshold
                filtered_paa_list = non_branded_paa_list[non_branded_paa_list['Similarity'] >= similarity_threshold]
                filtered_paa_list['Similarity'] = (filtered_paa_list['Similarity'] * 100).round().astype(int)


                columns_to_display = ['Question', 'Similarity']
                filtered_paa_list_to_display = filtered_paa_list[columns_to_display]

                # Display the filtered DataFrame without the index
                st.table(filtered_paa_list_to_display.reset_index(drop=True))
            else:
                st.error("paa_list is not structured as a DataFrame. Please check its format.")
            

        st.session_state['comparison_chart_data'] = comparison_chart_data
        if 'chart_data' in st.session_state:
            st.subheader('Top 50 Words - Average TF and TF-IDF Scores')
            chart_data = st.session_state['chart_data'].set_index('Terms')
            st.bar_chart(chart_data, color=["#FFAA00", "#6495ED","#FF5511"])


        # **Visualization: Bar and Line Chart**
        st.subheader('Comparison of Average TF and Your Content TF Scores')

        # Create the bar and line chart using Altair
        base = alt.Chart(filtered_chart_data).encode(
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
        tab1, tab2, tab3 = st.tabs(["Word Frequency", "Edit Terms", "LDA Terms"])
        with tab1:
            # Update sidebar label
            st.markdown("<div style='text-align: center; font-size: 24px; color: #ffaa00;'>Word Frequency</div>", unsafe_allow_html=True)
            st.markdown("<div style='padding: 1px; background-color: #f8f9fa; border-radius: 15px;'>", unsafe_allow_html=True)
            if text_input.strip():
                for idx, row in comparison_chart_data.iterrows():
                    keeping = row['Keep']
                    if keeping == True:
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
        with tab2:
            # Retrieve comparison_chart_data from st.session_state
            if 'comparison_chart_data' in st.session_state:
                comparison_chart_data = st.session_state['comparison_chart_data']
            else:
                st.error("No comparison chart data found. Please perform analysis first.")
                return

            if comparison_chart_data.empty:
                st.error("No terms available to edit.")
                return

            # Add 'Keep' column if it doesn't exist
            if 'Keep' not in comparison_chart_data.columns:
                comparison_chart_data['Keep'] = True
            edited_comparison_chart_data = pd.DataFrame()

            def update():
                for idx, change in st.session_state.terms["edited_rows"].items():
                    for label, value in change.items():
                        # st.write(f"{idx}  {label}  {value}")
                        st.session_state['comparison_chart_data'].loc[idx, label] = value
                        #st.table(comparison_chart_data)



            # Select columns to display and edit
            editable_columns = ['Keep', 'Terms']
            # Use st.data_editor to allow term editing
            edited_comparison_chart_data = st.data_editor(
                comparison_chart_data[editable_columns],
                column_config={
                    "Keep": st.column_config.CheckboxColumn(required=True),
                },
                use_container_width=True,
                key='terms',
                hide_index=True,
                on_change=update
            )
            comparison_chart_data.update(edited_comparison_chart_data)
            # Update comparison_chart_data in st.session_state
            st.session_state['comparison_chart_data'] = comparison_chart_data.copy()
        with tab3:
            st.markdown("<div style='text-align: center; font-size: 24px; color: #FFD700;'>LDA Terms</div>", unsafe_allow_html=True)
            st.markdown("<div style='padding: 1px; background-color: #f0f0f0; border-radius: 15px;'>", unsafe_allow_html=True)
            lda_top_terms = st.session_state.get('lda_top_terms', [])
            for term in lda_top_terms:
                st.markdown(f"<div style='padding: 8px; background-color: #FFD700; color: black; border-radius: 5px; margin-bottom: 5px;'>"
                            f"<strong>{term}</strong>"
                            f"</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
                # Tab 4: PAA Questions
        # with tab4:
        #     st.markdown("<div style='text-align: center; font-size: 24px; color: #555555;'>Edit PAA Questions</div>", unsafe_allow_html=True)

        #     if 'paa_list' not in st.session_state:
        #         st.session_state['paa_list'] = [
        #             {"Question": "What is the best way to improve SEO?", "Keep": True},
        #             {"Question": "How to optimize content for search engines?", "Keep": True},
        #             {"Question": "What are long-tail keywords?", "Keep": True},
        #         ]
        #     paa_list = st.session_state['paa_list']

        #     if 'Keep' not in paa_list.columns:
        #         paa_list['Keep'] = True
        #     edited_paa_df = pd.DataFrame()

        #     def update():
        #         for idx, change in st.session_state.terms["edited_rows"].items():
        #             for label, value in change.items():
        #                 # st.write(f"{idx}  {label}  {value}")
        #                 st.session_state['paa_list'].loc[idx, label] = value
        #                 #st.table(comparison_chart_data)

        #     editable_columns_q = ['Keep', 'Question']
        #     edited_paa_df = st.data_editor(
        #         paa_list[editable_columns_q],
        #         column_config={
        #             "Keep": st.column_config.CheckboxColumn(required=True),
        #         },
        #         use_container_width=True,
        #         key='Question',
        #         hide_index=True,
        #         on_change=update
        #     )
            
        #     paa_list.update(edited_paa_df)
            # Update `paa_df` in session state
            #st.session_state['paa_list'] = edited_paa_df.copy()

            # # Update `paa_list` in session state based on `Keep` column
            # st.session_state['paa_list'] = edited_paa_df[edited_paa_df['Keep']]['Question'].tolist()


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
