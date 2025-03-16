# Resume Enhancer and Job Application Recommendation System

## Project Overview

This project is a comprehensive web application built using **Streamlit** to help users optimize their resumes and identify suitable job roles. The system analyzes user-provided resumes and job descriptions (JD) to deliver actionable insights and recommendations. Key features include:

1. **Resume-JD Matching Score**:  
   - Implemented a talent matching system using **scikit-learn** and **Hugging Face Transformers** to compare resumes with job descriptions.  
   - Provides a compatibility score to assess candidate suitability for specific roles.

2. **Resume Enhancement Recommendations**:  
   - Offers tailored suggestions to improve resumes based on job descriptions.  
   - Focuses on ATS (Applicant Tracking System) optimization to increase visibility to recruiters.

3. **Job Role Recommendations**:  
   - Leverages **NLTK** and machine learning algorithms to recommend relevant job roles based on the user’s resume content.

4. **Resume Draft Generation**:  
   - Utilizes **Jinja2** to generate an initial resume draft customized to align with the provided job description.

## Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python (FastAPI), SQL (optional)  
- **Libraries**: PyPDF2, PDFMiner, NLTK, scikit-learn, Hugging Face Transformers, NumPy, Pandas  
- **Version Control**: Git/GitHub  

## Key Features

### Feature 1: Resume-JD Matching Score
The system compares the user's resume with the provided job description and generates a compatibility score to determine how well the resume aligns with the role.

### Feature 2: Resume Enhancement Recommendations
Based on the job description, the system provides actionable recommendations to improve the resume, ensuring it is optimized for ATS.

### Feature 3: Job Role Recommendations
The application suggests relevant job roles that align with the user's skills and experience, extracted from their resume.

### Feature 4: Resume Draft Generation
Generates a customized resume draft tailored to the job description using **Jinja2** templates.

## Future Enhancements
- Integrate live job posting recommendations to streamline the job application process further.
