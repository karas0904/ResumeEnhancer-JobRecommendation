import streamlit as st
import os
import tempfile
from model import extract_text_from_pdf, score_resume, enhance_resume

# Page title & icon
st.set_page_config(page_title="AI Resume Enhancer", page_icon="📄")

# Header with emoji
st.title("📄 AI Resume Enhancer")
st.write("Upload your resume and job description as PDFs, and let AI enhance your resume to match the job!")

# File upload section
st.subheader("Upload Resume & Job Description (PDF)")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📂 Upload Resume (PDF)", type=["pdf"])
with col2:
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
                #highlighted_resume = highlight_differences(resume_text, improved_resume)

                # Display with color formatting
                st.subheader("📌 Updated Resume ")
                st.markdown(f'<div style="border:1px solid #ddd; padding:10px; border-radius:5px;">{improved_resume}</div>', unsafe_allow_html=True)

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
