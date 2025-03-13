import time
import numpy as np
import pandas as pd
import requests
import base64
import math
import re
from typing import Optional, Any
import streamlit as st
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import nltk
import spacy
import ssl
import string
import os
import torch
import difflib
import people_also_ask as paa
from streamlit_quill import st_quill
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from supabase import create_client, Client
torch.classes.__path__ = [] # add this line to manually set it to empty. 
from google.oauth2 import service_account
from google.ads.googleads.client import GoogleAdsClient
import json
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.decomposition import PCA

from dotenv import load_dotenv

load_dotenv()


# For st_login_form, you can specify them here OR rely on environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")      # e.g. "https://<project>.supabase.co"
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_TABLE="users"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Attempt to fix SSL issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download needed NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
max_contents = 10

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        from spacy.cli import download
        download('en_core_web_sm')
        return spacy.load('en_core_web_sm')

nlp = load_spacy_model()

@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

ner_pipeline = load_ner_pipeline()
def compute_ner_count(text):
    """Compute the number of named entities in the text."""
    if not text:
        return 0
    entities = ner_pipeline(text[:2000])  # Limit to first 2000 characters for speed
    return len(entities)

def compute_pos_counts(text, normalize=True):
    """Count the number of adverbs, adjectives, and verbs in the given text.
    
    Args:
        text (str): The input text to analyze.
        normalize (bool): Whether to return normalized counts. Defaults to True.
    
    Returns:
        dict: A dictionary with counts of adverbs, adjectives, and verbs.
    """
    doc = nlp(text)
    total_words = len([token for token in doc if token.is_alpha])  # Exclude punctuation

    pos_counts = {
        "adverbs": sum(1 for token in doc if token.pos_ == "ADV"),
        "adjectives": sum(1 for token in doc if token.pos_ == "ADJ"),
        "verbs": sum(1 for token in doc if token.pos_ == "VERB")
    }

    if normalize and total_words > 0:
        for key in pos_counts:
            pos_counts[key] /= total_words  # Normalize each POS count
    elif normalize:
        for key in pos_counts:
            pos_counts[key] = 0  # Avoid division by zero

    return pos_counts

def compute_lexical_diversity(text):
    """Compute lexical diversity as the ratio of unique words to total words."""
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.is_alpha]
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0

def compute_readability_score(text: str) -> float:
    """
    Compute Flesch-Kincaid Grade using textstat.
    For very short texts (like a title), the score may be meaningless.
    """
    if not text or len(text.split()) < 3:
        return None
    return textstat.flesch_kincaid_grade(text)

def compute_text_stats_from_html(html_content: str):
    """
    Parse the HTML to extract paragraphs and compute various text statistics.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Find paragraphs by <p> tags
    paragraphs = soup.find_all("p")
    
    # Fallback if no <p> tags found (optional):
    if not paragraphs:
        # You could treat the entire HTML as a single "paragraph".
        paragraphs = [soup]

    paragraph_count = len(paragraphs)
    total_sentences = 0
    total_words = 0
    sentence_lengths = []

    for p in paragraphs:
        p_text = p.get_text(separator=" ", strip=True)
        # Tokenize into sentences
        sentences = sent_tokenize(p_text)
        total_sentences += len(sentences)

        # For each sentence, tokenize words to measure length
        for sent in sentences:
            words = word_tokenize(sent)
            sentence_lengths.append(len(words))
            total_words += len(words)

    # Avoid division by zero if the text is empty
    avg_sentences_per_paragraph = 0
    if paragraph_count > 0:
        avg_sentences_per_paragraph = total_sentences / paragraph_count

    avg_words_per_sentence = 0
    if total_sentences > 0:
        avg_words_per_sentence = total_words / total_sentences

    # Compute reading ease scores (requires `pip install textstat`)
    # Extract plain text from the entire HTML for textstat
    plain_text = soup.get_text(separator=" ", strip=True)
    flesch_reading_ease = textstat.flesch_reading_ease(plain_text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(plain_text)

    return {
        "paragraph_count": paragraph_count,
        "sentence_count": total_sentences,
        "word_count": total_words,
        "avg_sentences_per_paragraph": avg_sentences_per_paragraph,
        "avg_words_per_sentence": avg_words_per_sentence,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "sentence_lengths": sentence_lengths,
    }

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def optimal_kmeans_clusters(embeddings, max_k=10):
    """Find optimal number of clusters using elbow method"""
    distortions = []
    K = range(2, max_k+1)
    for k in K:
        kmean_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmean_model.fit(embeddings)
        distortions.append(kmean_model.inertia_)
    
    # Calculate the optimal k using elbow point detection
    deltas = np.diff(distortions)
    optimal_k = np.argmax(deltas < 0.5 * deltas[0]) + 2  # +2 for 0-based index and range starts at 2
    return optimal_k if optimal_k > 2 else 3

def get_cluster_names(df, embeddings, n_clusters):
    """Get representative cluster names using hybrid scoring"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_names = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = df['Cluster'] == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        
        # Calculate centroid
        centroid = cluster_embeddings.mean(axis=0)
        
        # Hybrid scoring
        scores = (
            0.6 * cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten() +
            0.2 * df[cluster_mask]['Impressions'] / 
                  df[cluster_mask]['Impressions'].max() +
            0.2 * (1 - df[cluster_mask]['Position'] / 
                         df[cluster_mask]['Position'].max())
        )
        
        best_idx = scores.argmax()
        cluster_names[cluster_id] = df[cluster_mask].iloc[best_idx]['Query']
    
    return cluster_names

def perform_kmeans_clustering(df, embeddings):
    """Perform KMeans clustering with interactive visualization"""
    # Determine optimal cluster count
    n_clusters = optimal_kmeans_clusters(embeddings)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    df['Cluster'] = clusters
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Get meaningful cluster names
    cluster_names = get_cluster_names(df, embeddings, n_clusters)
    df['Cluster Name'] = df['Cluster'].map(cluster_names)
    
    # Create interactive plot
    fig = px.scatter(
        df,
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        color='Cluster Name',
        hover_name='Query',
        hover_data={
            'Cluster Name': True,
            'Impressions': True,
            'CTR': ':.2f',
            'Position': True
        },
        title=f'Query Clusters (KMeans, {n_clusters} clusters)',
        labels={'color': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.G10,  # Use Prism color scale
    )
    
    # Enhance plot layout
    fig.update_layout(
        height=800,
        width=1200,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Add cluster centers
    centers = pca.transform(kmeans.cluster_centers_)
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='markers',
            marker=dict(
                color='black',
                size=12,
                symbol='x'
            ),
            name='Cluster Centers'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return df

def google_custom_search(query, api_key, cse_id, num_results=10, delay=1):
    """Performs Google Custom Search with rate-limiting."""
    all_results = []
    start_index = 1
    while len(all_results) < num_results:
        remaining = num_results - len(all_results)
        current_num = min(10, remaining)
        url = "https://customsearch.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': api_key,
            'cx': cse_id,
            'num': current_num,
            'start': start_index,
            'hl': 'en',
            'cr': 'countryUS'
        }
        try:    
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            items = data.get('items', [])
            if not items:
                break
            all_results.extend(items)
            start_index += current_num
            time.sleep(delay*10)  # delay between requests
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:
                print("Rate limit exceeded. Retrying in 60 seconds...")
                time.sleep(120)
                continue
            else:
                print(f"Error: {e}")
                return []
    return all_results


def extract_content_from_url(url, extract_headings=False, retries=2, timeout=5):
    """Extract main textual content (paragraphs) from a webpage."""
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/58.0.3029.110 Safari/537.3'
        )
    }
    for _ in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
                headings = []
                if extract_headings:
                    for level in ['h2', 'h3', 'h4']:
                        for tag in soup.find_all(level):
                            headings.append({'level': level, 'text': tag.get_text(strip=True)})

                icon_link = soup.find('link', rel=lambda x: x and ('icon' in x.lower()))
                if icon_link and icon_link.get('href'):
                    favicon_url = urljoin(url, icon_link['href'])
                else:
                    favicon_url = urljoin(url, '/favicon.ico')

                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
                return title, content.strip(), favicon_url, headings, soup
        except requests.RequestException:
            pass
        time.sleep(1)
    return None, "", "", [], None

def detailed_extraction(soup, url):
    # Clone the soup to avoid modifying the original
    soup_clone = BeautifulSoup(str(soup), 'html.parser')
    
    # Remove the <footer> element and its contents so we only analyze the article content
    footer = soup_clone.find("footer")
    if footer:
        footer.decompose()
    
    # Extract Title
    title = soup_clone.title.string.strip() if soup_clone.title and soup_clone.title.string else "No Title"
    
    # Extract Meta Description
    meta_description = None
    meta_tag = soup_clone.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_description = meta_tag.get("content").strip()
    
    # Extract main Content (using <p> tags) and count paragraphs
    paragraphs = soup_clone.find_all("p")
    content = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    num_paragraphs = len([p for p in paragraphs if p.get_text().strip()])
    
    # Count headings and bullet lists (only those above the removed footer)
    num_h2 = len(soup_clone.find_all("h2"))
    num_h3 = len(soup_clone.find_all("h3"))
    num_bullet_lists = len(soup_clone.find_all("ul"))

    return {
        "url": url,
        "title": title,
        "meta_description": meta_description,
        "content": content,
        "num_paragraphs": num_paragraphs,
        "num_h2": num_h2,
        "num_h3": num_h3,
        "num_bullet_lists": num_bullet_lists
    }

@st.cache_resource
def lemmatize_text(text: str) -> str:
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        # context-aware overrides
        if token.text.lower() == "media" and token.lemma_.lower() == "medium":
            lemmatized_tokens.append("media")
        elif token.text.lower() == "data" and token.lemma_.lower() == "datum":
            lemmatized_tokens.append("data")
        elif token.text.lower() == "publishers" and token.lemma_.lower() == "publisher":
            lemmatized_tokens.append("publisher")
        else:
            lemmatized_tokens.append(token.lemma_)
    return ' '.join(lemmatized_tokens)


def remove_duplicate_questions(questions, similarity_threshold=0.75):
    # If 0 or 1 questions, there's nothing to deduplicate
    if len(questions) < 2:
        return questions

    # Preprocess questions
    def preprocess(text):
        # Lowercase, remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    # Encode questions using SentenceTransformer
    model = load_embedding_model()
    preprocessed = [preprocess(q) for q in questions]
    embeddings = model.encode(preprocessed)

    # If embeddings is empty or only 1 row, again just return
    if embeddings.shape[0] < 2:
        return questions

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Cluster questions
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        linkage='complete',
        distance_threshold=1 - similarity_threshold
    )
    clustering_model.fit(1 - similarity_matrix)

    # Select a representative question from each cluster
    cluster_labels = clustering_model.labels_
    cluster_map = {}
    for idx, label in enumerate(cluster_labels):
        cluster_map.setdefault(label, []).append(questions[idx])

    final_questions = []
    for _, qs in cluster_map.items():
        # pick the shortest question from the cluster
        rep = min(qs, key=len)
        final_questions.append(rep)

    return final_questions


def extract_brand_name(url, title):
    parsed = urlparse(url)
    parts = parsed.netloc.split('.')
    if parts and parts[0] == 'www':
        parts.pop(0)
    domain_root = parts[0].capitalize() if parts else 'Unknown'

    if title:
        segs = title.split(' - ')
        for seg in reversed(segs):
            ratio = difflib.SequenceMatcher(None, domain_root.lower(), seg.lower()).ratio()
            if ratio > 0.8:
                return seg.strip()
    return domain_root

@st.cache_resource
def is_brand_mentioned(term, brand_name):
    # direct substring
    if brand_name.lower() in term.lower():
        return True
    # fuzzy match ratio
    ratio = difflib.SequenceMatcher(
        None,
        term.lower().replace(' ', ''),
        brand_name.lower().replace(' ', '')
    ).ratio()
    if ratio > 0.8:
        return True
    # check for named entity
    doc = nlp(term)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'PERSON', 'GPE']:
            ratio_ent = difflib.SequenceMatcher(
                None,
                ent.text.lower().replace(' ', ''),
                brand_name.lower().replace(' ', '')
            ).ratio()
            if ratio_ent > 0.8:
                return True
    return False


def is_not_branded(question):
    """Return True if question does NOT mention any brand in st.session_state['brands']"""
    brands = st.session_state.get('brands', [])
    for brand in brands:
        if is_brand_mentioned(question, brand):
            return False
    return True

# Initialize sentiment pipeline (cache as needed)
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def compute_readability(text):
    if not text or len(text.split()) < 3:
        return None
    try:
        return textstat.flesch_kincaid_grade(text)
    except Exception:
        return None

def compute_sentiment(text):
    text = text.strip()
    if not text:
        return None
    try:
        # If text is very long, limit to first 512 characters
        trimmed_text = text if len(text) <= 512 else text[:512]
        result = sentiment_pipeline(trimmed_text)
        if result:
            label = result[0]['label'].upper()
            score = result[0]['score']
            return score if label == "POSITIVE" else -score
        else:
            return None
    except Exception:
        return None

def compute_serp_features(details, position):
    content = details.get("content", "")
    pos_counts = compute_pos_counts(content)  # Get POS counts

    features = {
        "position": position,
        "url": details.get("url"),
        "title": details.get("title"),
        "title_readability": compute_readability(details.get("title")),
        "title_sentiment": compute_sentiment(details.get("title")),
        "meta_readability": compute_readability(details.get("meta_description") or ""),
        "meta_sentiment": compute_sentiment(details.get("meta_description") or ""),
        "content_readability": compute_readability(details.get("content")),
        "content_sentiment": compute_sentiment(details.get("content")),
        "word_count": len(details.get("content", "").split()),
        "num_paragraphs": details.get("num_paragraphs"),
        "num_h2": details.get("num_h2"),
        "num_h3": details.get("num_h3"),
        "num_bullet_lists": details.get("num_bullet_lists"),
        "entity_count": compute_ner_count(content),
        "lexical_diversity": compute_lexical_diversity(content),
        "adverbs": pos_counts["adverbs"],
        "adjectives": pos_counts["adjectives"],
        "verbs": pos_counts["verbs"]
    }
    return features

import os
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.oauth2 import service_account
from math import ceil
from textwrap import wrap

def get_keyword_plan_data(keywords_list=None, url=None, seed_mode="KW"):
    # 1) Read the JSON secret, replace newlines if needed
    creds_json = st.secrets['GOOGLE_SERVICE_ACCOUNT_JSON_CONTENT']
    creds_json = creds_json.replace("\n", "\\n")  # ensure valid JSON
    creds_dict = json.loads(creds_json)

    kw = st.session_state['keyword']
    model = load_embedding_model()
    keyword_embedding = model.encode([kw])[0]
    
    # 2) Create the service_account Credentials
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    
    # 3) Instantiate the Google Ads client
    client = GoogleAdsClient(
        credentials=credentials,
        developer_token="p2Of8yLD6yKNWn7NrtlR3g",
        login_customer_id="8882181823",
    )
    
    st.write("Client successfully initialized")

    # 4) Use the client
    keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
    google_ads_service = client.get_service("GoogleAdsService")

    language_rn = google_ads_service.language_constant_path("1000")  # English
    all_keyword_ideas = []
    if seed_mode=="URL":
        # Get keyword ideas from URL seed
        request.url_seed.url = url
        keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        for idx, idea in keyword_ideas.results:
            if len(all_keyword_ideas) > 400:
                break  # Changed from keyword_ideas.keyword_idea to keyword_ideas.results
            metrics = idea.keyword_idea_metrics
            comp_enum = client.enums.KeywordPlanCompetitionLevelEnum.KeywordPlanCompetitionLevel
            comp_label = comp_enum.Name(metrics.competition)
            comp_index = getattr(metrics, "competition_index", 0)

            idea_embeddings = model.encode(idea.text)
            similarity = cosine_similarity(
                keyword_embedding.reshape(1, -1),
                idea_embeddings.reshape(1, -1)
            )[0][0]

            all_keyword_ideas.append({
                "Keyword Text": idea.text,
                "Average Monthly Searches": metrics.avg_monthly_searches or 0,
                "Competition": comp_label,
                "Competition Index": comp_index,
                "Similarity to Keyword": similarity,
                "Word Count": len(idea.text.split()),
                "Position": idx + 1,
            })
        # Calculate max_avg_monthly_searches before the loop
        max_avg_monthly_searches = max((idea["Average Monthly Searches"] for idea in all_keyword_ideas), default=1)  # Avoid division by zero

        for idea in all_keyword_ideas:
            # Ensure "Average Monthly Searches" is not None before division
            avg_monthly_searches = idea["Average Monthly Searches"] if idea["Average Monthly Searches"] is not None else 0

            idea["Total Score"] = (
            0.1 * avg_monthly_searches / max_avg_monthly_searches +
            0.4 * idea["Similarity to Keyword"] +
            0.5 * (1 - math.sqrt(idea["Position"] / len(all_keyword_ideas)))
            )

    elif seed_mode=="KW":
        # Basic chunking approach
        CHUNK_SIZE = 20
        num_chunks = ceil(len(keywords_list) / CHUNK_SIZE)

        for i in range(num_chunks):
            chunk = keywords_list[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
            request = client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = "8882181823"
            request.language = language_rn
            request.include_adult_keywords = False
            request.keyword_plan_network = (
                client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
            )
            request.keyword_seed.keywords.extend(chunk)

            keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
            for idea in keyword_ideas:
                if len(all_keyword_ideas) > 400:
                    break  # Changed from keyword_ideas.keyword_idea to keyword_ideas.results
                
                metrics = idea.keyword_idea_metrics
                comp_enum = client.enums.KeywordPlanCompetitionLevelEnum.KeywordPlanCompetitionLevel
                comp_label = comp_enum.Name(metrics.competition)
                comp_index = getattr(metrics, "competition_index", 0)

                idea_embeddings = model.encode(idea.text)
                similarity = cosine_similarity(
                    keyword_embedding.reshape(1, -1),
                    idea_embeddings.reshape(1, -1)
                )[0][0]
                all_keyword_ideas.append({
                    "Keyword Text": idea.text,
                    "Average Monthly Searches": metrics.avg_monthly_searches or 0,
                    "Competition": comp_label,
                    "Competition Index": comp_index,
                    "Similarity to Keyword": similarity
                })
    else:
        st.error("Invalid seed mode selected.")
        return None
    df = pd.DataFrame(all_keyword_ideas).drop_duplicates().reset_index(drop=True)
    return df

def create_correlation_radar(target, pearson_corr, spearman_corr):
    """
    Generate a radar chart comparing Pearson and Spearman correlations with the target variable.
    """
    # Remove the target itself from the features list
    features = [f for f in pearson_corr.columns if f != target]
    if not features:
        st.info("No features available for radar chart.")
        return
    
    # Get correlations for each feature (dropping the target)
    pearson_vals = pearson_corr[target].drop(target).values
    spearman_vals = spearman_corr[target].drop(target).values
    
    N = len(features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the circle for plotting
    angles += angles[:1]
    pearson_vals = np.concatenate([pearson_vals, [pearson_vals[0]]])
    spearman_vals = np.concatenate([spearman_vals, [spearman_vals[0]]])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.plot(angles, pearson_vals, color="r", linewidth=2, label="Pearson")
    ax.fill(angles, pearson_vals, color="r", alpha=0.25)
    ax.plot(angles, spearman_vals, color="b", linewidth=2, label="Spearman")
    ax.fill(angles, spearman_vals, color="b", alpha=0.25)
    
    # Set the feature labels around the circle
    ax.set_xticks(angles[:-1])
    # Optionally wrap long feature names
    wrapped_labels = ["\n".join(wrap(label, 10)) for label in features]
    ax.set_xticklabels(wrapped_labels, fontsize=10)
    
    ax.set_title(f"Radar Chart: Correlations with {target}", y=1.1, fontsize=14, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    st.pyplot(fig)

def extract_headings_for_paa_list(top_urls):
    """Extract headings for PAA list from the given top URLs."""
    headings_data = []

    for idx, url in enumerate(top_urls):
        print(f"\nProcessing URL {idx+1}/{len(top_urls)} for headings: {url}")
        t, _, _, heads, _ = extract_content_from_url(url, extract_headings=True)
        if t is None:
            t = "No Title"

        if heads:
            for h in heads:
                if 'text' in h:
                    headings_data.append({
                        'text': h['text'].strip(),
                        'url': url,
                        'title': t
                    })
        time.sleep(0.5)

    if headings_data:
        question_words = ['how', 'why', 'what', 'who', 'which', 'is', 'are', 'can', 'does', 'will']
        # headings that either end in ? or start with a question word
        filtered_headings_data = [
            h for h in headings_data
            if (h['text'].endswith('?')
               or (h['text'].split() and h['text'].split()[0].lower() in question_words))
        ]
        # remove duplicates by (text, url, title)
        unique_set = {(hd['text'], hd['url'], hd['title']) for hd in filtered_headings_data}
        filtered_headings_data = [
            {'text': t, 'url': u, 'title': ti}
            for (t,u,ti) in unique_set
        ]
        # fetch PAA
        google_paa = []
        # remove duplicates across PAA

        # Combine them into a DataFrame
        if len(filtered_headings_data) > 0:
            df_hd = pd.DataFrame(filtered_headings_data, columns=['text','url','title'])
            paa_rows = [{'Question': q, 'URL': 'No URL', 'Title': 'No Title'} for q in google_paa]
            # Convert headings to same columns
            heading_rows = df_hd.rename(columns={'text':'Question','url':'URL','title':'Title'})
            # combine
            paa_df = pd.concat([heading_rows, pd.DataFrame(paa_rows)], ignore_index=True)

            st.session_state['paa_list'] = paa_df
            # Keep a separate headings_df for detailed headings
        else:
            # fallback empty
            st.session_state['paa_list'] = pd.DataFrame(columns=['Question','URL','Title'])
    else:
        st.session_state['paa_list'] = pd.DataFrame(columns=['Question','URL','Title'])


def display_serp_details():
    st.header("SERP Details")
    st.write("Analyze the SERP in more detail.")
    print("Displaying SERP Details")
    # Safety check: ensure the analysis was run
    if 'serp_contents' not in st.session_state or not st.session_state['serp_contents']:
        st.warning("No SERP content available. Please run the analysis first.")

    # Extract headings for PAA list
    if 'successful_urls' in st.session_state:
        extract_headings_for_paa_list(st.session_state['successful_urls'])

    # 1) Process each SERP entry: extract detailed data and compute features
    features_list = []
    serp_data = st.session_state['serp_contents']
    for row in serp_data:
        # Use the soup object and URL stored in each entry.
        details = detailed_extraction(row['soup'], row['url'])
        # Compute features for this SERP entry using its position.
        feat = compute_serp_features(details, row['position'])
        features_list.append(feat)

    # 2) Create a DataFrame for analysis and display
    df = pd.DataFrame(features_list)
    st.subheader("Computed Features for Each URL")
    st.dataframe(df)

    # --- New Section: Google Keyword Planner API ---
    st.subheader('Google Keyword Planner Data')
    if 'keyword' in st.session_state:
        keyword = st.session_state['keyword']
        df_kw_ideas = get_keyword_plan_data(keywords_list=[keyword], url=None, seed_mode="KW")
        if not df_kw_ideas.empty:
            st.dataframe(df_kw_ideas)
        else:
            st.warning("No keyword data retrieved from Google Keyword Planner.")

    # 3) Visualize distributions for each numeric metric
    st.subheader("Distributions of Numeric Metrics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        chart = alt.Chart(df).mark_bar().encode(
            alt.X(f"{col}:Q", bin=alt.Bin(maxbins=15), title=col),
            alt.Y("count()", title="Frequency")
        ).properties(
            width=300,
            height=200,
            title=f"Distribution of {col}"
        )
        st.altair_chart(chart, use_container_width=True)

    # 4) Display a correlation heatmap for the numeric features
    st.subheader("Correlation Matrix Heatmap")
    if numeric_cols:
        corr = df[numeric_cols].corr().reset_index().melt("index")
        chart_corr = alt.Chart(corr).mark_rect().encode(
            x=alt.X("variable:N", title="Feature"),
            y=alt.Y("index:N", title="Feature"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
            tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")]
        ).properties(width=500, height=500)
        st.altair_chart(chart_corr, use_container_width=True)
    else:
        st.info("No numeric data available for correlation analysis.")

    # 5) Example scatter plot: Word Count vs. Content Readability
    st.subheader("Scatter Plot: Word Count vs. Content Readability")
    if "word_count" in df.columns and "content_readability" in df.columns:
        scatter = alt.Chart(df).mark_circle(size=100).encode(
            x=alt.X("word_count:Q", title="Word Count"),
            y=alt.Y("content_readability:Q", title="Content Readability (Flesch-Kincaid Grade)"),
            color=alt.Color("position:O", title="SERP Position", scale=alt.Scale(scheme="purplered")),
            tooltip=[
                alt.Tooltip("url:N", title="URL"),
                alt.Tooltip("word_count:Q", title="Word Count"),
                alt.Tooltip("content_readability:Q", title="Content Readability"),
                alt.Tooltip("position:O", title="SERP Position")
            ]
        ).properties(width=500, height=400)
        st.altair_chart(scatter, use_container_width=True)

    # POS Counts - Adverbs, Adjectives, Verbs
    st.subheader("Parts of Speech (POS) Analysis")

    pos_melted = df.melt(id_vars=["position"], value_vars=["adverbs", "adjectives", "verbs"],
                            var_name="POS", value_name="Count")

    pos_chart = alt.Chart(pos_melted).mark_bar().encode(
        x=alt.X("position:O", title="SERP Position"),
        y=alt.Y("Count:Q", title="POS Count"),
        color=alt.Color("POS:N", title="Part of Speech"),
        tooltip=[alt.Tooltip("POS:N", title="Part of Speech"), alt.Tooltip("Count:Q", title="Count")]
    ).properties(width=600, height=400)
    
    st.altair_chart(pos_chart, use_container_width=True)
    st.subheader("Scatter Plots: POS Proportions vs. Readability")

    # 3) Grouped Bar Chart: Average POS Counts Across Readability Levels
    st.subheader("Average POS Usage Across Readability Levels")

    readability_bins = pd.cut(df["content_readability"], bins=[0, 5, 10, 15, 20, 25], labels=["0-5", "5-10", "10-15", "15-20", "20+"])
    df["readability_group"] = readability_bins

    avg_pos_df = df.groupby("readability_group", observed=True)[["adverbs", "adjectives", "verbs"]].mean().reset_index().melt(id_vars=["readability_group"], var_name="POS", value_name="Average Usage")

    avg_pos_chart = alt.Chart(avg_pos_df).mark_bar().encode(
        x=alt.X("readability_group:N", title="Readability Score Range"),
        y=alt.Y("Average Usage:Q", title="Average POS Usage"),
        color=alt.Color("POS:N", title="Part of Speech"),
        tooltip=[alt.Tooltip("POS:N", title="Part of Speech"), alt.Tooltip("Average Usage:Q", title="Usage", format=".4f")]
    ).properties(width=650, height=400)
    
    st.altair_chart(avg_pos_chart, use_container_width=True)

    # Compute similarity of questions to the keyword
    if 'paa_list' in st.session_state and 'keyword' in st.session_state:
        st.subheader("PAA Questions Similarity to Keyword")
        paa_df = st.session_state['paa_list']
        keyword = st.session_state['keyword']
        model = load_embedding_model()
        keyword_embedding = model.encode([keyword])[0]
        paa_embeddings = model.encode(paa_df['Question'].tolist())
        similarities = cosine_similarity([keyword_embedding], paa_embeddings)[0]
        paa_df['Similarity'] = similarities

        # Filter out branded questions and questions with less than 4 words
        paa_df = paa_df[paa_df['Question'].apply(is_not_branded)]
        paa_df = paa_df[paa_df['Question'].apply(lambda x: len(x.split()) >= 4)]
        
        # Compute NER count for each question
        paa_df['NER Count'] = paa_df['Question'].apply(compute_ner_count)
        
        # Filter out rows with similarity below 0.15 or NER count above 2
        paa_df = paa_df[(paa_df['Similarity'] >= 0.18) & (paa_df['NER Count'] <= 2)]
        
        st.dataframe(paa_df[['Question', 'Similarity', 'NER Count']])

    # 6) Button to return to the Editor screen
    if st.button("Return to Editor"):
        st.session_state['step'] = 'editor'
        st.rerun()



# streamlit_app.py (or a file that handles your app screens)
from gscHelpers import connect_to_search_console, load_gsc_query_data, load_gsc_query_data_alt

from streamlit_oauth import OAuth2Component

def display_gsc_analytics():
    st.title("Google Search Console Analysis")

    if 'CLIENT_ID' in st.secrets:
        CLIENT_ID = st.secrets['CLIENT_ID']
    elif 'GCS_CLIENT_ID2' in st.secrets:
        CLIENT_ID = st.secrets['GCS_CLIENT_ID2']
    else:
        st.error("No Google Search Console client ID found.")
        return
    if 'CLIENT_SECRET' in st.secrets:
        CLIENT_SECRET = st.secrets['CLIENT_SECRET']
    elif 'GCS_CLIENT_SECRET' in st.secrets:
        CLIENT_SECRET = st.secrets['GCS_CLIENT_SECRET2']
    else:
        st.error("No Google Search Console client secrets found.")
        return
    AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/auth"
    TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
    REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"
    SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]

    # OAuth logic
    if "auth" not in st.session_state:
        st.write("Not authenticated yet. Please log in via Google below:")
        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
        result = oauth2.authorize_button(
            name="Continue with Google",
            icon="https://www.google.com.tw/favicon.ico",
            redirect_uri="https://needsmorewords.streamlit.app/component/streamlit_oauth.authorize_button",
            scope="https://www.googleapis.com/auth/webmasters.readonly", #https://www.googleapis.com/auth/webmasters.readonly	
            key="google",
            extras_params={"prompt": "consent", "access_type": "offline"},
            use_container_width=True,
            pkce='S256',
        )
        if result:
            # Store tokens in session state
            st.session_state["token"] = result["token"]
            st.write(result)
            if "id_token" in result["token"]:
                id_token = result["token"]["id_token"]
                payload = id_token.split(".")[1]
                payload += "=" * (-len(payload) % 4)
                user_info = json.loads(base64.b64decode(payload))
                email = user_info.get("email", "No email found")
            else:
                id_token = result["token"]["access_token"]
                email = "GSC@USER"

            st.session_state["auth"] = email
            st.rerun()

    else:
        st.write(f"Logged in")
        if st.button("Logout"):
            del st.session_state["auth"]
            del st.session_state["token"]
            st.rerun()

        # Retrieve tokens from session
        if "token" in st.session_state:
            token = st.session_state["token"]
            access_token = token["access_token"]
            refresh_token = token.get("refresh_token", None)

            # Connect to GSC
            search_console_service = connect_to_search_console(
                access_token, refresh_token,
                CLIENT_ID, CLIENT_SECRET,
                SCOPES
            )

            if search_console_service:
                st.success("Connected to GSC! Now you can list your sites or query data.")
                
                # List Websites (i.e., site URL properties)
                if st.button("List My Websites"):
                    try:
                        response = search_console_service.sites().list().execute()
                        site_entries = response.get('siteEntry', [])
                        if site_entries:
                            st.write("Your GSC Websites:")
                            for s in site_entries:
                                st.write(f"• {s['siteUrl']}")

                            # Optionally store the first site in session
                            st.session_state["site_url"] = site_entries[0]['siteUrl']
                        else:
                            st.warning("No websites found in your Search Console account.")
                    except Exception as e:
                        st.error(f"Error listing websites: {e}")

                # If a site is already chosen, let user pick date range & fetch queries
                chosen_site = st.session_state.get("site_url", None)
                if chosen_site:
                    st.write(f"Selected site: {chosen_site}")

                    # Date inputs
                    start_date = st.date_input("Start Date", value=pd.to_datetime('2025-01-01'))
                    end_date = st.date_input("End Date", value=pd.to_datetime('2025-03-01'))

                    # Check if GSC data is already in session state
                    if "gsc_query_df" in st.session_state:
                        df = st.session_state["gsc_query_df"]
                        st.write("Using cached Query Data:")
                    else:
                        if st.button("Get Query Data"):
                            # Clear the screen
                            st.empty()

                            df = load_gsc_query_data(
                                service=search_console_service,
                                site_url=chosen_site,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d')
                            )
                            if not df.empty:
                                # Store the full DF in session_state
                                st.session_state["gsc_query_df"] = df
                            else:
                                st.warning("No query data was returned.")
                                return  # Exit if no data is returned
                        else:
                            return  # Wait for the button to be pressed

                    df_gsc = df.copy()
                    st.subheader("Underperforming Keywords Analysis")

                    # ----- (A) Basic aggregator approach -----
                    # For demonstration, let's define a few bins for 'position'
                    # and compute the average CTR for each bin as a reference:
                    position_bins = [0, 1, 3, 6, 10, 100]
                    bin_labels = ["pos1", "pos2-3", "pos4-6", "pos7-10", "pos10+"]
                    df_gsc["position_bin"] = pd.cut(df_gsc["Position"], bins=position_bins, labels=bin_labels)
                    bin_ctr = df_gsc.groupby("position_bin", observed=True)["CTR"].mean()

                    # Filter to queries with enough impressions to matter, e.g. >= 100
                    df_filtered = df_gsc[df_gsc["Impressions"] >= 100].copy()

                    def underperforming(row):
                        """
                        Mark a query as 'underperforming' if CTR < 50% of the bin's average CTR.
                        """
                        bin_label = row["position_bin"]
                        if pd.isnull(bin_label):
                            return False
                        baseline = bin_ctr.get(bin_label, np.nan)
                        if np.isnan(baseline):
                            return False
                        if row["Position"] > 25:
                            return False
                        return row["CTR"] < 0.5 * baseline

                    df_filtered["Is_Underperforming"] = df_filtered.apply(underperforming, axis=1)

                    # Merge Is_Underperforming into df_gsc
                    df_gsc = df_gsc.merge(df_filtered[["Query", "Is_Underperforming"]], on="Query", how="left")
                    df_gsc["Is_Underperforming"] = df_gsc["Is_Underperforming"].fillna(False, downcast='infer')
                    # Sort by Impressions descending
                    df_underperf = df_filtered[df_filtered["Is_Underperforming"]].sort_values("Impressions", ascending=False)
                    
                    if not df_underperf.empty:
                        st.write("Underperforming Queries (CTR < 50% of average for their position):")

                        # Define column configuration
                        column_configuration = {
                            "Query": st.column_config.TextColumn(disabled=True),
                            "Impressions": st.column_config.NumberColumn(disabled=True),
                            "CTR": st.column_config.NumberColumn(disabled=True),
                            "Position": st.column_config.NumberColumn(disabled=True),
                            "Is_Underperforming": st.column_config.CheckboxColumn(disabled=True),
                        }

                        # Display the DataFrame with multi-row selection
                        event = st.dataframe(
                            df_underperf,
                            column_config=column_configuration,
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="multi-row",
                        )

                        # Get selected rows
                        selected_rows = event.selection.rows

                        # Filter the DataFrame based on selected rows
                        filtered_df = df_underperf.iloc[selected_rows]

                        # Display selected rows
                        st.header("Selected Underperforming Queries")
                        st.dataframe(
                            filtered_df,
                            column_config=column_configuration,
                            use_container_width=True,
                            hide_index=True
                        )
                        # Generate embeddings for clustering
                        # model = load_embedding_model()
                        # embeddings = model.encode(df_gsc.head(300)['Query'].tolist())

                        # # Perform clustering and visualization
                        # clustered_df = perform_kmeans_clustering(df_gsc.head(300), embeddings)
                        st.header("Cannibalized Queries Analysis")
                        dimensions_query_page = ["page"]
                        query_page_df = load_gsc_query_data(
                                service=search_console_service,
                                site_url=chosen_site,
                                start_date=start_date.strftime('%Y-%m-%d'),
                                end_date=end_date.strftime('%Y-%m-%d'),
                                dimensions=dimensions_query_page
                            )

                        if not query_page_df.empty:
                            st.subheader("Raw Data with Query and Page Dimensions")
                            query_page_df = query_page_df.sort_values("Clicks", ascending=False)
                            max_page_queries = 20
                            query_page_df = query_page_df.head(max_page_queries)
                            page_urls = query_page_df['Query'].unique()
                            
                            all_query_page_data = pd.DataFrame()
                            for i, page_url in enumerate(page_urls):
                                if i >= max_page_queries:
                                    break
                                info_placeholder = st.empty()  # Create a placeholder
                                info_placeholder.info(f"Fetching data for page: {page_url}")
                                dimensions_query = ["query"] # We only need 'query' dimension now, page is filtered
                                query_df = load_gsc_query_data_alt(
                                    service=search_console_service,
                                    site_url=chosen_site,
                                    start_date=start_date.strftime('%Y-%m-%d'),
                                    end_date=end_date.strftime('%Y-%m-%d'),
                                    page_add=page_url,
                                    dimensions=dimensions_query
                                )
                                query_df['Page'] = page_url
                                
                                # info_placeholder.write(f"Data fetched for page: {query_df}")
                                time.sleep(1)
                                info_placeholder.empty()  # Clear the info message

                                if not query_df.empty:
                                    all_query_page_data = pd.concat([all_query_page_data, query_df]) # Append page data to the combined DataFrame
                                else:
                                    st.info(f"No data returned for page: {page_url}")

                        if not all_query_page_data.empty:
                            st.subheader("Raw Data (Page-by-Page Queries)")
                            st.dataframe(all_query_page_data)

                            if st.button("Analyze Keyword Cannibalization", key="cannibalization_analysis"):
                                # Identify cannibalized queries (queries associated with multiple pages)
                                query_page_grouped = all_query_page_data.groupby('Query')['Page'].nunique().reset_index()
                                query_page_grouped.columns = ['Query', 'Unique_Page_Count']
                                cannibalized_queries_df = query_page_grouped[query_page_grouped['Unique_Page_Count'] > 1]

                                if not cannibalized_queries_df.empty:
                                    st.subheader("Cannibalized Queries")
                                    st.dataframe(cannibalized_queries_df)

                                    # Merge cannibalized queries with original data to get performance metrics
                                    cannibalized_queries_metrics_df = pd.merge(
                                        cannibalized_queries_df,
                                        df_gsc,
                                        on='Query',
                                        how='inner'
                                    )

                                    st.subheader("Performance Metrics for Cannibalized Queries")
                                    st.dataframe(cannibalized_queries_metrics_df)

                                    # --- Visualizations ---
                                    st.subheader("Visualizations")

                                    # 1. Bar chart of number of cannibalized queries
                                    fig_cannibalized_count = px.bar(
                                        x=['Cannibalized Queries', 'Non-Cannibalized Queries'],
                                        y=[len(cannibalized_queries_df), len(query_page_grouped) - len(cannibalized_queries_df)],
                                        title="Number of Cannibalized vs. Non-Cannibalized Queries",
                                        labels={'y': 'Number of Queries', 'x': 'Query Type'}
                                    )
                                    st.plotly_chart(fig_cannibalized_count)

                                    # 2. Table of cannibalized queries and their page counts (already displayed as dataframe)

                                    # 3. Performance metrics summary for cannibalized queries
                                    cannibalized_summary_metrics = cannibalized_queries_metrics_df[[ 'Clicks', 'Impressions', 'CTR', 'Position']].mean().reset_index()
                                    cannibalized_summary_metrics.columns = ['Metric', 'Average Value']
                                    st.dataframe(cannibalized_summary_metrics)

                                    all_queries_summary_metrics = query_page_df[[ 'Clicks', 'Impressions', 'CTR', 'Position']].mean().reset_index()
                                    all_queries_summary_metrics.columns = ['Metric', 'Average Value']

                                    summary_metrics_comparison = pd.DataFrame({
                                        'Metric': all_queries_summary_metrics['Metric'],
                                        'All Queries': all_queries_summary_metrics['Average Value'].round(2),
                                        'Cannibalized Queries': cannibalized_summary_metrics['Average Value'].round(2) if not cannibalized_summary_metrics.empty else [0,0,0,0]
                                    })

                                    st.subheader("Comparison of Average Performance Metrics")
                                    st.dataframe(summary_metrics_comparison)

                                    # 4. Distribution of unique page counts per query
                                    fig_page_count_distribution = px.histogram(
                                        query_page_grouped,
                                        x='Unique_Page_Count',
                                        title="Distribution of Unique Page Counts per Query",
                                        labels={'Unique_Page_Count': 'Number of Unique Pages'}
                                    )
                                    st.plotly_chart(fig_page_count_distribution)


                                    st.subheader("Detailed List of Cannibalized Queries and Pages")
                                    expanded_cannibalized_data = pd.merge(
                                        cannibalized_queries_df['Query'],
                                        query_page_df,
                                        on='Query',
                                        how='inner'
                                    )
                                    st.dataframe(expanded_cannibalized_data)

                                else:
                                    st.info("No cannibalized queries found in the selected date range.")

                        else:
                            st.info("No query data found for the selected date range.")

                    else:
                        st.info("No underperforming queries found.")

                    # Show a scatter plot: Position vs. CTR, bubble sized by Impressions
                    st.markdown("### Position vs CTR (All Queries)")

                    # Create chart with modified color encoding
                    chart = alt.Chart(df_gsc).mark_circle().encode(
                        x=alt.X("Position:Q", title="Position"),
                        y=alt.Y("CTR:Q", title="CTR"),
                        size=alt.Size("Impressions:Q", scale=alt.Scale(range=[10,400])),
                        color=alt.Color('Is_Underperforming:N',
                            scale=alt.Scale(
                                domain=[True, False],
                                range=['coral', 'teal']
                            )
                        ),
                        tooltip=["Query", "Impressions", "CTR", "Position", "Is_Underperforming"]
                    ).properties(width=700, height=400).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)

                    # ----- Return button -----
                    if st.button("Return to Main"):
                        st.session_state['step'] = 'analysis'
                        st.rerun()
                else:
                    st.error("Failed to connect to GSC with given credentials.")

def filter_terms(terms):
    """Filter out numeric, stopword, or other low-value tokens."""
    custom_stopwords = set([
        "i","me","my","myself","we","our","ours","ourselves","you","your","way","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
        "it","its","itself","they","them","their","theirs","themselves","what","which","who",
        "whom","this","that","these","those","am","is","are","was","were","be","been","being",
        "have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or",
        "because","as","until","while","of","at","by","for","with","about","against","between","into",
        "through","during","before","after","above","below","to","from","up","down","in","out","on","off",
        "over","under","again","further","then","once","here","there","when","where","why","how","all",
        "any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same",
        "so","than","too","very","s","t","can","will","just","don","should","now","like","need"
    ])

    filtered = []
    seen = set()
    for term in terms:
        if any(ch.isdigit() for ch in term):
            continue
        doc = nlp(term)
        # skip if it has undesired POS
        if any(tok.pos_ in ['AUX','PRON','DET','ADP','CCONJ','NUM','SYM','PUNCT'] for tok in doc):
            continue
        # check for stopwords
        lemmas = [t.lemma_.lower() for t in doc]
        if any(lm in custom_stopwords or lm in nlp.Defaults.stop_words for lm in lemmas):
            continue
        final_lemma = ' '.join(lemmas)
        if final_lemma not in seen:
            filtered.append(final_lemma)
            seen.add(final_lemma)
    return filtered


def perform_analysis(keyword):
    """Refactored function using logic consistent with the React+FastAPI version,
       but preserving EXACT st.session_state keys and formats used in the original code.
    """
    start_time = time.time()
    user_email = st.session_state["username"] or "guest" # or "user_id" if you have it

    status_placeholder = st.empty()  # Create a placeholder for dynamic updates
    status_placeholder.info('Retrieving top search results...')

    st.session_state['keyword'] = keyword
    st.session_state['serp_contents'] = []  # NEW: store structured data

    api_key = os.getenv("API_KEY")
    cse_id = os.getenv("CSE_ID")

    # 1) Retrieve search items
    results = google_custom_search(keyword, api_key, cse_id, num_results=35)
    if not results:
        status_placeholder.error('No results found.')
        return

    top_urls = [item['link'] for item in results if 'link' in item]
    if not top_urls:
        status_placeholder.error('No URLs found.')
        return
    st.session_state['top_urls'] = top_urls

    # 2) Extract content from top URLs
    titles, favicons, retrieved_content = [], [], []
    headings_data = []
    successful_urls = []
    word_counts = []
    brand_names = set()

    progress = st.progress(0)
    for idx, url in enumerate(top_urls):
        if len(retrieved_content) >= max_contents:
            break
        print(f"\nProcessing URL {idx+1}/{len(top_urls)}: {url}")
        progress.progress(idx / len(top_urls))
        status_placeholder.info(f"Retrieving content from {url}...")
        t, content, favicon_url, heads, soup = extract_content_from_url(url, extract_headings=True)
        if t is None:
            t = "No Title"

        # brand
        print("Filtering branded content")
        brand_name = extract_brand_name(url, t)
        brand_names.add(brand_name)

        if heads:
            for h in heads:
                if 'text' in h:
                    headings_data.append({
                        'text': h['text'].strip(),
                        'url': url,
                        'title': t
                    })

        if content:
            wc = len(content.split())
            retrieved_content.append(content)
            successful_urls.append(url)
            titles.append(t)
            favicons.append(favicon_url)
            # anchor word count at least 1000
            word_counts.append(wc if wc > 1000 else 1000)
            st.session_state['serp_contents'].append({
                "position": idx + 1,      # 1-based SERP rank
                "url": url,
                "title": t,
                "content": content,
                "favicon": favicon_url,
                "word_counts": wc if wc > 1000 else 1000,
                "soup": soup
            })
        time.sleep(0.5)
    st.session_state['successful_urls'] = successful_urls
    progress.empty()
    status_placeholder.empty()  # Remove the last message after completion
    # store brand names
    st.session_state['brands'] = list(brand_names)

    if not retrieved_content:
        st.error('Failed to retrieve sufficient content.')
        return

    if len(word_counts) > 0:
        ideal_count = int(np.median(word_counts)) + 500
    else:
        ideal_count = 1000
    st.session_state['ideal_word_count'] = ideal_count

    # 3) Clean and lemmatize
    docs_lemmatized = [lemmatize_text(doc) for doc in retrieved_content]

    # 5) Display top search results
    st.subheader('Top Search Results')
    for i in range(len(titles)):
        fc = favicons[i]
        t = titles[i]
        link = successful_urls[i]
        wc = word_counts[i]
        st.markdown(
            f"""
            <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: black;">
                <div style="display: flex; align-items: center;">
                    <img src="{fc}" width="32" style="margin-right: 10px;">
                    <div>
                        <strong>{t}</strong> ({wc} words)<br>
                        <a href="{link}" target="_blank">{link}</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    lower_bound = (ideal_count // 500) * 500
    upper_bound = lower_bound + 500
    st.success(f"**Suggested Word Count:** Aim for approx. {lower_bound}–{upper_bound} words based on top content.")

    print("Starting TF-IDF operations")

    # 6) TF-IDF + CountVectorizer
    model = load_embedding_model()
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
    tf_vectorizer    = CountVectorizer(ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs_lemmatized).toarray()
    tf_matrix    = tf_vectorizer.fit_transform(docs_lemmatized).toarray()

    feature_names = tfidf_vectorizer.get_feature_names_out()
    filtered_feats = filter_terms(feature_names)

    # filter the matrices
    idxs = [i for i, term in enumerate(feature_names) if term in filtered_feats]
    tfidf_matrix_f = tfidf_matrix[:, idxs]
    tf_matrix_f    = tf_matrix[:, idxs]
    filtered_feature_names = [feature_names[i] for i in idxs]

    # compute average
    avg_tfidf = np.mean(tfidf_matrix_f, axis=0)
    avg_tf    = np.mean(tf_matrix_f, axis=0)
    # also track doc lengths
    doc_word_counts = [len(d.split()) for d in docs_lemmatized]
    avg_doc_len = float(sum(doc_word_counts)) / max(1, len(doc_word_counts))

    # normalizing
    avg_tfidf /= avg_doc_len
    avg_tf    /= avg_doc_len

    print("Generating embeddings...")
    st.info("Indexing results... Generating embeddings...")
    # 7) Now compute similarity for each term to the user keyword
    keyword_emb = model.encode([keyword])[0]
    term_embeddings = model.encode(filtered_feature_names)
    similarities = cosine_similarity([keyword_emb], term_embeddings)[0]

    # "Combined Score" = average tf-idf * similarity
    combined_scores = avg_tfidf * similarities

    # get top 50
    N = 50
    top_idx = np.argsort(combined_scores)[-N:][::-1]
    top_terms = [filtered_feature_names[i] for i in top_idx]
    top_combined = [combined_scores[i] for i in top_idx]
    top_tfidf    = [avg_tfidf[i] for i in top_idx]
    top_tf       = [avg_tf[i] for i in top_idx]
    top_sim      = [similarities[i] for i in top_idx]

    # 8) Store in session_state
    # chart_data => DataFrame with columns: Terms, Combined Score, Average TF-IDF Score, Similarity to Keyword
    st.session_state['chart_data'] = pd.DataFrame({
        'Terms': top_terms,
        'Combined Score': top_combined,
        'Average TF-IDF Score': [x * 100 for x in top_tfidf],
        'Similarity to Keyword': [x * 100 for x in top_sim]
    })
    print(top_tfidf)

    # words_to_check => list of dicts with Term, Average TF Score, Average TF-IDF Score
    st.session_state['words_to_check'] = [
        {
            'Term': top_terms[i],
            'Average TF Score': top_tf[i],
            'Average TF-IDF Score': top_tfidf[i]
        }
        for i in range(len(top_terms))
    ]
    # data_to_store = {
    #     "keyword": keyword,
    #     "top_urls": top_urls,  # or successful_urls
    #     "brand_names": list(brand_names),
    #     "ideal_word_count": ideal_count,
    #     # Combine your relevant data (like words_to_check, chart_data, etc.) in JSON
    #     "analysis_data": {
    #         "words_to_check": st.session_state['words_to_check'],
    #         "chart_data": st.session_state['chart_data'].to_dict() if 'chart_data' in st.session_state else {},
    #         # add more fields as needed...
    #     },
    # }
    # try:
    #     response = supabase.table("analysis_results").insert({
    #         "user_email": user_email,
    #         "keyword": keyword,
    #         "top_urls": top_urls,
    #         "brand_names": list(brand_names),
    #         "ideal_word_count": ideal_count,
    #         "analysis_data": data_to_store['analysis_data']
    #     }).execute()

    # except Exception as e: #Exception?
    #     st.error(f"Supabase insert failed: {e}")
    # # else:
    # inserted_rows = response.data
    # if inserted_rows and len(inserted_rows) > 0:
    #     new_id = inserted_rows[0]["id"]
    #     st.session_state["analysis_id"] = new_id
    #     st.success(f"Saved analysis data with ID: {new_id}")


    st.session_state['analysis_completed'] = True

    elapsed_time = time.time() - start_time
    print(f"Time taken for analysis: {elapsed_time:.2f} seconds")

def display_editor():
    # Add a button to start a new analysis
    if st.button('Start a New Analysis'):
        st.session_state.clear()
        st.session_state["authenticated"] = False
        st.rerun()

    # Retrieve the ideal word count from session state
    ideal_word_count = st.session_state.get('ideal_word_count', None)

    # ---- NEW BUTTON to see SERP details ----
    if st.button("Analyze SERP in Detail", key='serp_details',type="primary", icon=":material/zoom_in:"):
        st.session_state['step'] = 'serp_details'
        st.rerun()

    # Display the ideal word count suggestion
    if ideal_word_count:
        lower_bound = (ideal_word_count // 500) * 500
        upper_bound = lower_bound + 500
        st.info(f"**Suggested Word Count:** Aim for approximately {lower_bound} to {upper_bound} words based on top-performing content.")

    # Update sidebar label
    st.sidebar.subheader('Optimize Your Content with These Words')

    # Grab words_to_check from session_state
    words_to_check = st.session_state['words_to_check']

    # Create a Quill editor for inputting and editing text
    text_input = st_quill(placeholder='Start typing your content here...', key='quill')

    # Adjust Quill editor height
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

    if text_input is None:
        text_input = ""

    # Remove HTML tags
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text_input, 'html.parser')
    text_input_plain = soup.get_text()

    # Lemmatize the user input
    text_input_lemmatized = lemmatize_text(text_input_plain)

    # Calculate the word count dynamically
    word_count = len(text_input_plain.split()) if text_input_plain.strip() else 0

    # --- New Section: Import SEO Keywords ---
    st.subheader('Import SEO Keywords')
    uploaded_file = st.file_uploader('Upload a CSV file from your SEO keyword research tool:', type='csv')

    imported_keywords = None
    if uploaded_file is not None:
        df_seo = pd.read_csv(uploaded_file)
        df_seo = df_seo[['Keyword', 'Avg. Search Volume (Last 6 months)', 'Keyword Difficulty']]
        df_seo['Avg. Search Volume (Last 6 months)'] = pd.to_numeric(df_seo['Avg. Search Volume (Last 6 months)'], errors='coerce')
        df_seo['Keyword Difficulty'] = pd.to_numeric(df_seo['Keyword Difficulty'], errors='coerce')
        df_seo['Score'] = df_seo['Avg. Search Volume (Last 6 months)'] / (df_seo['Keyword Difficulty'] + 1e-6)
        df_seo = df_seo.sort_values(by='Score', ascending=False)
        df_seo = df_seo.head(5).reset_index(drop=True)

        st.session_state['imported_keywords'] = df_seo
        imported_keywords = df_seo
    else:
        imported_keywords = st.session_state.get('imported_keywords', None)

    # Merge imported keywords with existing words_to_check
    tfidf_terms_set = set(word['Term'] for word in words_to_check)
    imported_words_to_check = []
    if imported_keywords is not None and not imported_keywords.empty:
        max_search_volume = imported_keywords['Avg. Search Volume (Last 6 months)'].max()
        for idx, row in imported_keywords.iterrows():
            term = row['Keyword']
            if term in tfidf_terms_set:
                continue
            else:
                search_volume = row['Avg. Search Volume (Last 6 months)']
                difficulty = row['Keyword Difficulty']
                weight = 3
                if pd.notna(search_volume) and max_search_volume > 0:
                    weight += (search_volume / max_search_volume) * 2
                else:
                    weight += 1
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
        imported_keywords = pd.DataFrame()  # empty fallback

    combined_words_to_check = imported_words_to_check + words_to_check

    if text_input_plain.strip():
        # Retrieve or build comparison_chart_data
        if 'comparison_chart_data' in st.session_state:
            comparison_chart_data = st.session_state['comparison_chart_data'].copy()
        else:
            comparison_chart_data = pd.DataFrame({
                'Terms': [w['Term'] for w in combined_words_to_check],
                'Average TF Score': [w.get('Average TF Score', 0) for w in combined_words_to_check],
                'IsImported': [w.get('IsImported', False) for w in combined_words_to_check],
                'Weight': [w.get('Weight', 1) for w in combined_words_to_check],
                'Keep': True
            })

        from sklearn.feature_extraction.text import CountVectorizer
        tf_vectorizer = CountVectorizer(vocabulary=comparison_chart_data['Terms'].tolist(), ngram_range=(1, 3))
        text_tf_matrix = tf_vectorizer.transform([text_input_lemmatized]).toarray()
        total_words = word_count

        # Editor TF Score
        editor_tf_scores = (text_tf_matrix[0] / total_words) * 1000 if total_words > 0 else np.zeros(len(comparison_chart_data))
        comparison_chart_data['Editor TF Score'] = editor_tf_scores

        # Occurrences
        import re
        occurrences_list = []
        for term in comparison_chart_data['Terms']:
            lem_term = lemmatize_text(term)
            occurrences = len(re.findall(r'\b' + re.escape(lem_term) + r'\b', text_input_lemmatized, flags=re.IGNORECASE))
            occurrences_list.append(occurrences)
        comparison_chart_data['Occurrences'] = occurrences_list

        # Targets and ranges
        targets = []
        deltas = []
        for idx, row in comparison_chart_data.iterrows():
            if row['IsImported']:
                target = max(1, int(word_count / 500))
                delta = max(1, int(0.1 * target))
            else:
                # Same logic you have now, but multiplying by the raw word_count or by 1?
                target = max(1, int(np.floor(row['Average TF Score'] * word_count)))
                delta = max(1, int(0.1 * target))
            targets.append(target)
            deltas.append(delta)
        comparison_chart_data['Target'] = targets
        comparison_chart_data['Delta'] = deltas
        comparison_chart_data['Min Occurrences'] = np.maximum(1, comparison_chart_data['Target'] - comparison_chart_data['Delta'])
        comparison_chart_data['Max Occurrences'] = comparison_chart_data['Target'] + comparison_chart_data['Delta']

        # Keep filter
        filtered_chart_data = comparison_chart_data[comparison_chart_data['Keep'] == True].reset_index(drop=True)

        # Optimization Score
        def compute_term_score(occ, tgt, mn, mx):
            if mn <= occ <= mx:
                range_score = 1
            else:
                range_score = 0
            proximity_score = max(0, 1 - abs(occ - tgt) / tgt)
            return 0.5 * range_score + 0.5 * proximity_score

        def compute_optimization_score(df):
            term_scores = []
            for _, r in df.iterrows():
                occ = r['Occurrences']
                tgt = r['Target']
                mn  = r['Min Occurrences']
                mx  = r['Max Occurrences']
                w   = r['Weight']
                ts  = compute_term_score(occ, tgt, mn, mx)
                term_scores.append(ts * w)
            total_weight = df['Weight'].sum()
            if total_weight <= 0:
                return 0
            score = (sum(term_scores) / total_weight) * 100
            score = max(score, 19) + 10
            score = min(score, 94)
            return score

        optimization_score = compute_optimization_score(filtered_chart_data)

        # Display word count + score
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

        st.session_state['comparison_chart_data'] = comparison_chart_data.copy()

        # PAA similarity, charting, etc.  (unchanged from your snippet)
        # ----------------------------------------------------------------

        # (Optional) charting code for your TF data...
        if 'chart_data' in st.session_state:
            st.subheader('Top 50 Words - Average TF and TF-IDF Scores')
            chart_data = st.session_state['chart_data'].set_index('Terms')
            st.bar_chart(chart_data, color=["#FFAA00", "#6495ED","#FF5511"])

        # etc...
        """
        Display additional or improved data visualizations using the data
        stored in st.session_state['chart_data'].
        """
        if 'chart_data' not in st.session_state:
            st.warning("No chart data found. Run the analysis first.")
            return

        chart_data = st.session_state['chart_data'].copy()
        # chart_data columns: ['Terms', 'Combined Score', 'Average TF-IDF Score', 'Similarity to Keyword']

        st.subheader("Additional Visualizations")

        # 1) Scatter (Bubble) Chart: TF-IDF vs. Similarity, bubble sized by Combined Score
        st.markdown("#### Scatter Chart: Average TF-IDF vs. Similarity to Keyword")
        scatter_chart = alt.Chart(chart_data).mark_circle().encode(
            x=alt.X('Average TF-IDF Score:Q'),
            y=alt.Y('Similarity to Keyword:Q'),
            size=alt.Size('Combined Score:Q', scale=alt.Scale(range=[30, 300])),
            color=alt.value('#ff2200'),
            tooltip=['Terms:N', 
                    alt.Tooltip('Combined Score:Q', format='.2f'), 
                    alt.Tooltip('Average TF-IDF Score:Q', format='.2f'), 
                    alt.Tooltip('Similarity to Keyword:Q', format='.2f')]
        ).interactive().properties(
            width=700,
            height=400
        )
        st.altair_chart(scatter_chart, use_container_width=True)

        # 2) Word Cloud
        st.markdown("#### Word Cloud of Top Terms")
        # Generate a simple dictionary of {term: combined_score} 
        term_weights = dict(zip(chart_data['Terms'], chart_data['Combined Score']))

        # You can tweak the word cloud params (width, height, background_color, etc.)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=50,
            colormap='Set1'  # Use a reddish theme
        ).generate_from_frequencies(term_weights)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        # from streamlit_oauth import OAuth2Component

        # # import logging
        # # logging.basicConfig(level=logging.INFO)

        # st.title("Google OIDC Example")
        # st.write("This example shows how to use the raw OAuth2 component to authenticate with a Google OAuth2 and get email from id_token.")

        # # create an OAuth2Component instance
        # CLIENT_ID = st.secrets["GOOGLE_CLIENT_ID"]
        # CLIENT_SECRET = st.secrets["GOOGLE_CLIENT_SECRET"]
        # AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/auth"
        # TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
        # REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"


        # if "auth" not in st.session_state:
        #     # create a button to start the OAuth2 flow
        #     oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT, REVOKE_ENDPOINT)
        #     result = oauth2.authorize_button(
        #         name="Continue with Google",
        #         icon="https://www.google.com.tw/favicon.ico",
        #         redirect_uri="https://needsmorewords.streamlit.app/component/streamlit_oauth.authorize_button",
        #         scope="https://www.googleapis.com/auth/webmasters.readonly", #https://www.googleapis.com/auth/webmasters.readonly	
        #         key="google",
        #         extras_params={"prompt": "consent", "access_type": "offline"},
        #         use_container_width=True,
        #         pkce='S256',
        #     )
            
        #     if result:
        #         st.write(result)
        #         # decode the id_token jwt and get the user's email address
        #         id_token = result["token"]["id_token"]
        #         # verify the signature is an optional step for security
        #         payload = id_token.split(".")[1]
        #         # add padding to the payload if needed
        #         payload += "=" * (-len(payload) % 4)
        #         payload = json.loads(base64.b64decode(payload))
        #         email = payload["email"]
        #         st.session_state["auth"] = email
        #         st.session_state["token"] = result["token"]
        #         st.rerun()
        # else:
        #     st.write("You are logged in!")
        #     st.write(st.session_state["auth"])
        #     st.write(st.session_state["token"])
        #     if st.button("Logout"):
        #         del st.session_state["auth"]
        #         del st.session_state["token"]
    else:
        st.markdown(f"Word Count: {word_count}")
    # ------------------------------------------------------------------
    # NEW HELPER: classify each term's target into a bucket
    # (like 1–3, 4–10, 11–20, 21–30, or 31+)
    # ------------------------------------------------------------------
    def classify_target(target):
        if target <= 3:
            return "1-3"
        elif target <= 10:
            return "3-10"
        elif target <= 20:
            return "1--20"
        elif target <= 30:
            return "20-30"
        else:
            return "30+"

    # Display words to check in the sidebar
    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["Word Frequency", "Edit Terms", "Text Stats"])

        # ---------------------------
        # Tab 1: Word Frequency
        # ---------------------------
        with tab1:
            st.markdown("<div style='text-align: center; font-size: 24px; color: #ffaa00;'>Word Frequency</div>", unsafe_allow_html=True)
            st.markdown("<div style='padding: 1px; background-color: #f8f9fa; border-radius: 15px;'>", unsafe_allow_html=True)

            # Only display if there is text
            if text_input_plain.strip():
                # Use the updated 'comparison_chart_data'
                comparison_chart_data = st.session_state.get('comparison_chart_data', pd.DataFrame())

                for idx, row in comparison_chart_data.iterrows():
                    if row['Keep'] is not True:
                        continue  # skip unchecked terms

                    term = row['Terms']
                    occurrences = row['Occurrences']
                    min_occ = row['Min Occurrences']
                    max_occ = row['Max Occurrences']
                    tgt = row['Target']

                    # 1) Determine color
                    if occurrences < min_occ:
                        color = "#E3E3E3"  # Light Gray
                    elif occurrences > max_occ:
                        color = "#EE2222"  # Red
                    else:
                        color = "#b0DD7c"  # Green

                    # 2) Different styling for imported
                    if row['IsImported']:
                        background_style = 'background-color: #E6FFE6;'  # Light green
                    else:
                        background_style = f'background-color: {color};'

                    # 3) Classify the target into a range
                    target_range_label = classify_target(tgt)

                    # 4) Show item in the sidebar
                    st.markdown(
                        f"""
                        <div style='display: flex; flex-direction: column; margin-bottom: 5px; 
                                    padding: 8px; {background_style} color: black; border-radius: 5px;'>
                            <span style='font-weight: bold;'>{term}</span>
                            <span>Occurrences: {occurrences} / Target: {target_range_label}</span>
                            <span style='font-size: 12px; color: #555;'>To be more precise: {min_occ}–{max_occ}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # 5) Progress bar (optional)
                    if min_occ > 0:
                        progress_val = min(1.0, occurrences / min_occ)
                    else:
                        progress_val = 1.0 if occurrences > 0 else 0
                    st.progress(progress_val)

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------
        # Tab 2: Edit Terms
        # ---------------------------
        with tab2:
            # Your existing editing code...
            if 'comparison_chart_data' in st.session_state:
                comparison_chart_data = st.session_state['comparison_chart_data']
            else:
                st.error("No comparison chart data found. Please perform analysis first.")
                return

            if comparison_chart_data.empty:
                st.error("No terms available to edit.")
                return

            if 'Keep' not in comparison_chart_data.columns:
                comparison_chart_data['Keep'] = True
            edited_comparison_chart_data = pd.DataFrame()

            def update():
                for idx, change in st.session_state.terms["edited_rows"].items():
                    for label, value in change.items():
                        st.session_state['comparison_chart_data'].loc[idx, label] = value

            editable_columns = ['Keep', 'Terms']
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
            st.session_state['comparison_chart_data'] = comparison_chart_data.copy()

        # ---------------------------
        # Tab 3: LDA Terms (unchanged)
        # ---------------------------
        with tab3:
            st.markdown("<h4 style='text-align: center;'>Text Statistics</h4>", unsafe_allow_html=True)

            # Only compute/show stats if there's content
            if text_input.strip():
                stats = compute_text_stats_from_html(text_input)
                pos_counts = compute_pos_counts(text_input, normalize=False)
                
                st.write("**Paragraph Count:**", stats["paragraph_count"])
                st.write("**Sentence Count:**", stats["sentence_count"])
                st.write("**Word Count:**", stats["word_count"])
                st.write("**Avg Sentences per Paragraph:**", f"{stats['avg_sentences_per_paragraph']:.2f}")
                st.write("**Avg Words per Sentence:**", f"{stats['avg_words_per_sentence']:.2f}")
                st.write("**Flesch Reading Ease:**", f"{stats['flesch_reading_ease']:.2f}")
                st.write("**Flesch-Kincaid Grade:**", stats['flesch_kincaid_grade'])
                st.write("**Adverbs Count:**", pos_counts["adverbs"])
                st.write("**Adjectives Count:**", pos_counts["adjectives"])
                st.write("**Verbs Count:**", pos_counts["verbs"])

                # Visualize distribution of sentence lengths (words per sentence)
                sentence_lengths = stats["sentence_lengths"]
                if sentence_lengths:
                    st.markdown("**Distribution of Sentence Lengths**")
                    # Create a DataFrame for Altair
                    length_df = pd.DataFrame({"sentence_length": sentence_lengths})

                    # A simple histogram
                    chart = (
                        alt.Chart(length_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("sentence_length:Q", bin=alt.Bin(maxbins=10), title="Words per Sentence"),
                            y=alt.Y("count()", title="Frequency")
                        )
                        .properties(width=300, height=200)
                    )
                    st.altair_chart(chart, use_container_width=True)
                    if st.button("Save Text Metrics"):
                        # Make sure we have an analysis_id
                        analysis_id = st.session_state.get("analysis_id", None)
                        if not analysis_id:
                            st.warning("No analysis ID found. Please run analysis and insert data first.")
                        else:
                            user_email = st.session_state["username"]
                            response = supabase.table("text_metrics").insert({
                                "user_email": user_email,
                                "analysis_id": analysis_id,
                                "word_count": stats["word_count"],
                                "paragraph_count": stats["paragraph_count"],
                                "sentence_count": stats["sentence_count"],
                                "flesch_reading_ease": stats["flesch_reading_ease"],
                                "flesch_kincaid_grade": stats["flesch_kincaid_grade"],
                            }).execute()

                            if response.get("status_code") and 200 <= response["status_code"] < 300:
                                st.success("Text metrics saved to database!")
                            else:
                                st.error("Error saving text metrics to database.")
                                print(response)
                else:
                    st.info("No sentences found to generate a distribution.")                
            else:
                st.info("No text available. Start typing to see statistics.")

    # Finally, display any imported keywords info at the bottom
    with st.sidebar:
        if imported_keywords is not None and not imported_keywords.empty:
            st.markdown("<div style='text-align: center; font-size: 20px; color: #2E8B57;'>Imported SEO Keywords</div>", unsafe_allow_html=True)
            for idx, row in imported_keywords.iterrows():
                kw = row['Keyword']
                sv = row['Avg. Search Volume (Last 6 months)']
                diff = row['Keyword Difficulty']
                occurrences = text_input.lower().count(kw.lower())
                st.markdown(f"""
                    <div style='padding: 8px; background-color: #E6FFE6; color: black; border-radius: 5px; margin-bottom: 5px;'>
                        <strong>{kw}</strong><br>
                        Occurrences: {occurrences}<br>
                        Avg. Search Volume (6 months): {sv}<br>
                        Keyword Difficulty: {diff}
                    </div>
                """, unsafe_allow_html=True)