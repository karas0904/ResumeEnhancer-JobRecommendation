import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# Page title & icon
st.set_page_config(page_title="AI Resume Enhancer", page_icon="📄")


import os
import tempfile
from LLM_For_Resume_Enhancement.model  import extract_text_from_pdf, score_resume, enhance_resume
from Live_Job_Postings.live_postings import fetch_jobs

# Load the model globally (outside any function) to avoid reloading
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Add this after your existing imports
@st.cache_data
def load_job_data():
    # Load your job postings data
    postings_sample_df = pd.read_csv("../Ml_Model_For_Similar_Jobs/postings.csv")  # Update path
    # Convert stored embeddings back to numpy arrays
    job_postings_embeddings = np.array([eval(emb) for emb in postings_sample_df['embeddings']])
    return postings_sample_df, job_postings_embeddings

# Add this function to your existing code
def get_similar_jobs(resume_text, postings_df, job_embeddings, top_n=5):
    # Generate embedding for the resume
    resume_embedding = model.encode([resume_text], show_progress_bar=False)
    
    # Calculate similarities
    similarities = cosine_similarity(resume_embedding, job_embeddings)
    
    # Get top matches
    top_indices = np.argsort(similarities[0])[::-1][:top_n]
    
    # Get matching jobs
    matches = []
    for idx in top_indices:
        job = postings_df.iloc[idx]
        similarity_score = similarities[0][idx]
        matches.append({
            'title': job['title'],
            'job_id': job['job_id'],
            'similarity': similarity_score
        })
    
    return matches



# Header with emoji
st.title("📄 AI Resume Enhancer")
st.write("Upload your resume and job description as PDFs, and let AI enhance your resume to match the job!")

# File upload section
st.subheader("Upload Resume & Job Description (PDF)")

resume_file = st.file_uploader("📂 Upload Resume (PDF)", type=["pdf"])

jd_file = st.file_uploader("📂 Upload Job Description (PDF)", type=["pdf"])


def save_uploaded_file(uploaded_file):
    """ Saves uploaded file temporarily and returns the path """
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
    return None

if st.button("🚀 Analyze Resume"):
    if resume_file and jd_file:
        with st.spinner("Extracting text from PDFs..."):
            resume_path = save_uploaded_file(resume_file)
            jd_path = save_uploaded_file(jd_file)

            resume_text = extract_text_from_pdf(resume_path)
            jd_text = extract_text_from_pdf(jd_path)

        if resume_text and jd_text:
            with st.spinner("🔍 Analyzing..."):
                score = score_resume(resume_text, jd_text)
                st.success(f"✅ Resume Match Score: **{score}/100**")

            with st.spinner("📝 Enhancing Resume..."):
                improved_resume = enhance_resume(resume_text, jd_text)
                

                # Display with color formatting
                st.subheader("📌 Updated Resume ")
                st.markdown(f'<div style="border:1px solid #ddd; padding:10px; border-radius:5px;">{improved_resume}</div>', unsafe_allow_html=True)

                st.subheader("Download the text file for viewing the changes in proper format")
                # Download Button
                st.subheader("📥 Download Updated Resume")
                st.download_button(label="📩 Download as .txt",
                                   data=improved_resume,
                                   file_name="Updated_Resume.txt",
                                   mime="text/plain")
        else:
            st.error("⚠️ Could not extract text from one or both PDFs. Try again.")
    else:
        st.error("⚠️ Please upload both Resume and Job Description PDFs.")


job_role = st.text_input("Job Role (e.g., 'Software Engineer')")


# Button to search for jobs
if st.button("Search Jobs"):
    if job_role:
        st.write(f"Searching for jobs as '{job_role}'...")
        jobs = fetch_jobs(job_role)
        
        if jobs:
            # Display the job listings
            for job in jobs:
                st.write(f"**Job Title:** {job.get('title')}")
                st.write(f"**Company:** {job.get('company', {}).get('display_name', 'N/A')}")
                st.write(f"**Location:** {job.get('location', {}).get('area', 'N/A')}")
                st.write(f"**Link:** [{job.get('redirect_url')}]({job.get('redirect_url')})")
                st.write("-" * 50)
        else:
            st.write("No jobs found.")
    else:
        st.warning("Please enter a job role to search.")
