import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import asyncio
import nest_asyncio
import tempfile
from LLM_For_Resume_Enhancement.model import extract_text_from_pdf, score_resume, enhance_resume
from Live_Job_Postings.live_postings import fetch_jobs
from Ml_Model_For_Similar_Jobs.job_role_recommendation import (
    get_top_matches_for_custom_resume,
    generate_embeddings_in_batches,
    validate_embeddings,
    combine_features
)


# Page config
st.set_page_config(page_title="AI Resume Enhancer", page_icon="📄")
nest_asyncio.apply()

# Load the model globally
@st.cache_resource
def load_model():
    try:
        # Set up event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return SentenceTransformer("all-MiniLM-L6-v2")
    except RuntimeError as e:
        st.error(f"Error loading model: {str(e)}")
        return None
    
@st.cache_data
def load_job_data():
    # Construct the file path relative to the script's directory
    file_path = os.path.join(os.path.dirname(__file__), "Ml_Model_For_Similar_Jobs", "postings.csv")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the CSV file
    try:
        postings_df = pd.read_csv(file_path)
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Unexpected error reading CSV file: {str(e)}")
        raise
    
    # Sample 1000 rows or use all if fewer are available
    if len(postings_df) < 1000:
        st.warning(f"CSV file has only {len(postings_df)} rows. Using all available rows.")
        postings_sample_df = postings_df
    else:
        postings_sample_df = postings_df.sample(1000)
    
    # Generate combined features and embeddings
    postings_sample_df['combined_features'] = postings_sample_df.apply(combine_features, axis=1)
    postings_sample_df['embeddings'] = list(generate_embeddings_in_batches(postings_sample_df, 'combined_features'))
    job_postings_embeddings = validate_embeddings(postings_sample_df['embeddings'].to_numpy())
    
    return postings_sample_df, job_postings_embeddings
# Initialize model and job data
model = load_model()
try:
    postings_sample_df, job_postings_embeddings = load_job_data()
except Exception as e:
    st.error(f"Error loading job data: {str(e)}")
    postings_sample_df, job_postings_embeddings = None, None

# Header
st.title("📄 AI Resume Enhancer")
st.write("Upload your resume and get personalized job recommendations and enhancements!")

# File upload section
st.subheader("Upload Resume & Job Description (PDF)")
resume_file = st.file_uploader("📂 Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("📂 Upload Job Description (PDF)", type=["pdf"])

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path
    return None

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Resume Enhancement", "Job Recommendations", "Job Search"])

with tab1:
    if st.button("🚀 Analyze Resume", key="analyze"):
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
                    st.subheader("📌 Updated Resume ")
                    st.markdown(f'<div style="border:1px solid #ddd; padding:10px; border-radius:5px;">{improved_resume}</div>', 
                              unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📩 Download as .txt",
                        data=improved_resume,
                        file_name="Updated_Resume.txt",
                        mime="text/plain"
                    )
            else:
                st.error("⚠️ Could not extract text from one or both PDFs. Try again.")
        else:
            st.error("⚠️ Please upload both Resume and Job Description PDFs.")

with tab2:
    if resume_file:
        if st.button("🎯 Get Job Recommendations", key="recommend"):
            with st.spinner("Finding matching jobs..."):
                resume_path = save_uploaded_file(resume_file)
                try:
                    matches = get_top_matches_for_custom_resume(
                        resume_path, 
                        job_postings_embeddings, 
                        postings_sample_df
                    )
                    
                    st.subheader("🎯 Top Job Matches")
                    for i, match in enumerate(matches, 1):
                        title, job_id, description, similarity = match
                        with st.expander(f"{i}. {title} (Score: {similarity:.2%})"):
                            st.write("**Job ID:**", job_id)
                            st.write("**Description:**", description)
                except Exception as e:
                    st.error(f"Error getting job recommendations: {str(e)}")
    else:
        st.warning("Please upload your resume to get job recommendations.")

with tab3:
    job_role = st.text_input("Job Role (e.g., 'Software Engineer')")
    if st.button("🔍 Search Jobs", key="search"):
        if job_role:
            with st.spinner(f"Searching for {job_role} positions..."):
                jobs = fetch_jobs(job_role)
                if jobs:
                    # Get first 5 unique jobs based on title and company
                    seen = set()
                    unique_jobs = []
                    for job in jobs:
                        key = (job.get('title'), job.get('company', {}).get('display_name'))
                        if key not in seen and len(unique_jobs) < 5:
                            seen.add(key)
                            unique_jobs.append(job)
                    
                    st.subheader("📋 Top 5 Job Matches")
                    for i, job in enumerate(unique_jobs, 1):
                        with st.expander(f"{i}. **{job.get('title')}** at {job.get('company', {}).get('display_name', 'N/A')}"):
                            st.write(f"**Location:** {job.get('location', {}).get('area', 'N/A')}")
                            st.write(f"**Link:** [{job.get('redirect_url')}]({job.get('redirect_url')})")
                else:
                    st.info("No jobs found for the specified role.")
        else:
            st.warning("Please enter a job role to search.")