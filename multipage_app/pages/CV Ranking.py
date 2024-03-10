import streamlit as st

st.title("CV Ranking")
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
from io import BytesIO
import PyPDF2

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from uploaded PDFs
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract emails and names using spaCy NER
def extract_entities(text):
    emails = re.findall(r'\S+@\S+', text)
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)
    if names:
        names = [" ".join(names[0])]
    return emails, names

# Function to process uploaded resumes
def process_resumes(uploaded_files):
    processed_resumes = []
    for file_name, content in uploaded_files.items():
        resume_text = extract_text_from_pdf(content)
        emails, names = extract_entities(resume_text)
        processed_resumes.append((file_name, names, emails, resume_text))
    return processed_resumes

# Function to rank resumes based on similarity
def rank_resumes(processed_resumes, job_description):
    tfidf_vectorizer = TfidfVectorizer()
    job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

    # Rank resumes based on similarity
    ranked_resumes = []
    for (file_name, names, emails, resume_text) in processed_resumes:
        resume_vector = tfidf_vectorizer.transform([resume_text])
        similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
        ranked_resumes.append((file_name, names, emails, similarity))

    # Sort resumes by similarity score
    ranked_resumes.sort(key=lambda x: x[3], reverse=True)

    return ranked_resumes

# Streamlit web app
def main():
    st.title("Resume Ranking Web App")

    # Job description input
    job_description = st.text_area("Enter Job Description:")

    # Upload resumes
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

    if st.button("Rank Resumes"):
        if not job_description or not uploaded_files:
            st.warning("Please enter a job description and upload resumes.")
        else:
            st.info("Ranking resumes...")

            # Process and rank resumes
            processed_resumes = process_resumes({file.name: file.getvalue() for file in uploaded_files})
            ranked_resumes = rank_resumes(processed_resumes, job_description)

            # Display results in a table
            st.write("Ranking Order:")
            st.table(
                {
                    "File Name": [result[0] for result in ranked_resumes],
                    "Names": [result[1] for result in ranked_resumes],
                    "Emails": [result[2] for result in ranked_resumes],
                    "Similarity Score": [result[3] for result in ranked_resumes],
                }
            )

if __name__ == "__main__":
    main()