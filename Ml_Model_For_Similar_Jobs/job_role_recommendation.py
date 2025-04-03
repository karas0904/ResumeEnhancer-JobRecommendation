import pandas as pd
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.probability import FreqDist
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import PyPDF2

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')

def preprocess_text(text):
    """Preprocess text by converting to lowercase, removing special characters, and handling NaN."""
    if pd.isnull(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

def combine_features(row):
    """Combine relevant features into a single string."""
    features = []
    for col in ['title', 'description', 'skills_desc']:
        if not pd.isnull(row[col]):
            features.append(f"{col.capitalize()}: {preprocess_text(row[col])}\n")
    return ' '.join(features)

def preprocess_resume_text(row):
    """Preprocess resume text by converting to lowercase, removing special characters, and handling NaN."""
    text = row.get('Resume_str', '')
    if pd.isnull(text):
        return ""
    text = re.sub(r'[^\w\s,+./-]', '', text)  # Remove unwanted characters but retain important ones 
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = text.strip()  # Trim leading/trailing whitespace
    text = text.lower() # Normalize to lowercase
    return text

def print_most_frequent_words(text, top_n=50):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    wordfreqdist = FreqDist(filtered_words)
    mostcommon = wordfreqdist.most_common(top_n)
    for word, freq in mostcommon:
        print(f"{word}: {freq}")

def generate_embeddings_in_batches(df, column_name, batch_size=32):
    embeddings = []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for i in tqdm(range(0, len(df), batch_size), desc="Generating Embeddings"):
        batch_texts = df[column_name].iloc[i:i + batch_size].tolist()
        try:
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error occurred during embedding generation: {e}")
            embeddings.append([np.nan] * len(batch_texts))
    return np.vstack(embeddings)

def validate_embeddings(embeddings):
    embeddings = np.array(embeddings, dtype=object)
    embedding_lengths = [len(embedding) if isinstance(embedding, (list, np.ndarray)) else 0 for embedding in embeddings]
    
    if len(set(embedding_lengths)) != 1:
        raise ValueError("All embeddings should have the same length. Found embeddings with differing lengths.")
    
    embeddings = np.array(embeddings.tolist(), dtype=np.float32)
    
    if embeddings.ndim != 2 or np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        print("Invalid embeddings found. Here are some problematic entries:")
        invalid_indices = np.where(np.isnan(embeddings) | np.isinf(embeddings))[0]
        print(embeddings[invalid_indices])
        embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
        embeddings = embeddings[~np.isinf(embeddings).any(axis=1)]
        if embeddings.ndim != 2:
            raise ValueError("Embeddings are not 2D after cleaning. Please check the input data.")
    
    return embeddings

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def get_top_matches_for_resume(resume_id, resume_df, job_postings_embeddings, postings_sample_df, top_n=5):
    if resume_id not in resume_df['ID'].values:
        raise ValueError(f"Resume ID {resume_id} not found in dataset.")
    
    resume_embedding = resume_df.loc[resume_df['ID'] == resume_id, 'resume_embeddings'].values[0]
    resume_embedding = np.array(resume_embedding, dtype=np.float32).reshape(1, -1)
    cosine_sim = cosine_similarity(resume_embedding, job_postings_embeddings)
    top_indices = np.argsort(cosine_sim[0])[::-1][:top_n]
    
    top_job_titles = postings_sample_df['title'].iloc[top_indices].values
    top_job_descriptions = postings_sample_df['description'].iloc[top_indices].values
    top_similarities = cosine_sim[0][top_indices]
    top_job_ids = postings_sample_df['job_id'].iloc[top_indices].values
    
    matches = list(zip(top_job_titles, top_job_ids, top_job_descriptions, top_similarities))
    return matches

def get_top_matches_for_custom_resume(pdf_path, job_postings_embeddings, postings_sample_df, top_n=5):
    resume_text = extract_text_from_pdf(pdf_path)
    if not resume_text:
        raise ValueError("No text found in the uploaded PDF. Please provide a valid resume.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_embedding = model.encode([resume_text], show_progress_bar=False)
    cosine_sim = cosine_similarity(resume_embedding, job_postings_embeddings)
    top_indices = np.argsort(cosine_sim[0])[::-1][:top_n]

    top_job_titles = postings_sample_df['title'].iloc[top_indices].values
    top_job_descriptions = postings_sample_df['description'].iloc[top_indices].values
    top_similarities = cosine_sim[0][top_indices]
    top_job_ids = postings_sample_df['job_id'].iloc[top_indices].values

    matches = list(zip(top_job_titles, top_job_ids, top_job_descriptions, top_similarities))
    return matches

def main():
    # Load datasets
    postings_df = pd.read_csv("postings.csv")
    resume_df = pd.read_csv("Resume.csv")
    
    # Process job postings
    postings_sample_df = postings_df.sample(1000)
    postings_sample_df['combined_features'] = postings_sample_df.apply(combine_features, axis=1)
    postings_sample_df['embeddings'] = list(generate_embeddings_in_batches(postings_sample_df, 'combined_features'))
    
    # Process resumes 
    resume_df['preprocessed_resume'] = resume_df.apply(preprocess_resume_text, axis=1)
    resume_df['resume_embeddings'] = list(generate_embeddings_in_batches(resume_df, 'preprocessed_resume'))
    
    # Validate embeddings
    resume_embeddings = validate_embeddings(resume_df['resume_embeddings'].tolist())
    job_postings_embeddings = validate_embeddings(postings_sample_df['embeddings'].to_numpy())
    
    # Example usage
    resume_id = 14206561
    matches = get_top_matches_for_resume(resume_id, resume_df, job_postings_embeddings, postings_sample_df)
    print(f"\nTop 5 job matches for Resume ID {resume_id}:")
    for i, match in enumerate(matches, start=1):
        title, job_id, description, similarity = match
        print(f"{i}. {title} (Job ID: {job_id}) - Similarity: {similarity:.4f}")

    # Example with custom PDF resume
    pdf_path = "VISHALRESUME-1.pdf"
    matches = get_top_matches_for_custom_resume(pdf_path, job_postings_embeddings, postings_sample_df)
    print(f"\nTop 5 job matches for uploaded resume:")
    for i, match in enumerate(matches, start=1):
        title, job_id, description, similarity = match
        print(f"{i}. {title} (Job ID: {job_id})")

if __name__ == "__main__":
    main()