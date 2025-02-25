import fitz  # PyMuPDF
import google.generativeai as genai
import re
import os
import difflib
from config import GEMINI_KEY


GENAI_API_KEY = GEMINI_KEY

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extracts and returns text content from a PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        return f"Error reading PDF: {e}"
    return text.strip()

def call_gemini(prompt):
    """Calls Gemini API and returns response."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text if response else "Error: No response from Gemini."

def score_resume(resume_text, jd_text):
    """Rates resume on a scale of 0-100 based on JD match."""
    prompt = f"""
    Evaluate the following resume against the job description and give a score from 0 to 100.

    Resume:
    {resume_text}

    Job Description:
    {jd_text}

    Output only: "Score: X" where X is the number.
    """
    response = call_gemini(prompt)
    match = re.search(r"Score:\s*(\d+)", response)
    return int(match.group(1)) if match else None

def enhance_resume(resume_text, jd_text):
    """Use Gemini API to enhance the resume based on JD with structured feedback."""
    prompt = f"""
    **You are a Resume Enhancer Agent** that provides precise suggestions to improve resumes based on job descriptions.

    **Input Files:**
    - **Resume:** {resume_text}
    - **Job Description:** {jd_text}

    **Task:**
    - Analyze the resume against the JD.
    - Provide an **updated resume** incorporating these improvements.
    - **Clearly mention changes in UPPER CASES** (e.g., **PYTHON(Suggestion)**).
    - **DO NOT** include any additional commentary or introduction.

    **Expected Output Format:**
    ```
    Skills:
    - PYTHON (Suggestion)
    - SQL, Tableau

    Work Experience:
    Data Analyst
    Fountain House
    May 2018 - Current / New York, NY
    - Built out the data and reporting infrastructure from the ground up using Tableau and SQL (Modified)
    - Designed A/B tests improving conversion by **19 basis points** (Impact added)
    - Led a team of **2 full-time employees and 4 contractors** (Modified)

    Certifications:
    - AWS CERTIFIED DATA ANALYST (Suggestions)
    ```

    **Note:**
    - Do not provide explanations beyond the structured response.
    - Ensure the updated resume flows naturally while incorporating changes.
    - Ensure to follow proper formating . Use seperate line break for different section of the resume. Use heading and subheadings as needed.
    """

    response = call_gemini(prompt)
    return response





