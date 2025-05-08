import streamlit as st
import requests
from LLM_For_Resume_Enhancement.config import APPLICATION_ID, APPLICATION_KEY

# Function to fetch job listings from Adzuna API
def fetch_jobs(job_role):
    url = f"https://api.adzuna.com/v1/api/jobs/in/search/1?app_id={APPLICATION_ID}&app_key={APPLICATION_KEY}&what={job_role}"
    response = requests.get(url)
    
    # Check if the API request was successful
    if response.status_code == 200:
        jobs = response.json().get('results', [])
        return jobs
    else:
        st.error(f"Error fetching jobs: {response.status_code}")
        return []
