# app.py
import streamlit as st
import requests
from config import APPLICATION_ID, APPLICATION_KEY

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

# Streamlit interface
st.title("Job Listing Finder")
st.write("Enter a job role to search for job listings:")

# Input box for job role
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
