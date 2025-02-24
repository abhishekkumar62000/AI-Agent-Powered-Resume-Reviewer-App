import streamlit as st
import ollama
import os
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader, PdfWriter
from docx import Document
from fpdf import FPDF

Resumes_path = "Resumes.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(Resumes_path)
except FileNotFoundError:
    st.sidebar.warning("Resumes.png file not found. Please check the file path.")
    
board_path = "board.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(board_path)
except FileNotFoundError:
    st.sidebar.warning("board.png file not found. Please check the file path.")


# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Sidebar configuration
with st.sidebar:
    st.header("⚙ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1-distill-llama-70b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🚨 Resume Expert
    - 🐞 Resume Assistant
    - 📝 Detail Documentation
    - 💡 Solution Design
    - 🚨 Job Description Check
    """)
    st.divider()
    st.markdown("👨👨‍💻Developer:- Abhishek❤️Yadav")
    
    developer_path = "my.jpg"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(developer_path)
except FileNotFoundError:
    st.sidebar.warning("my.jpg file not found. Please check the file path.")

def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return "Unsupported file format. Upload a PDF or DOCX."

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    vectors = FAISS.from_texts(chunks, embedding_model)
    return vectors

def analyze_resume(text):
    prompt = f"""
    You are an AI Resume Reviewer Develope by Abhishek from Bihar India. Analyze the following resume and provide feedback on:
    - Strengths
    - Areas of improvement
    - Formatting suggestions
    - Overall rating out of 10
    
    Resume:
    {text}
    """
    response = ollama.chat(model='deepseek-r1:7b', messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def match_job_description(resume_text, job_desc):
    prompt = f"""
    You are an AI Job Fit Analyzer  Develope by Abhishek from Bihar India. Compare the following resume with the given job description.
    Provide feedback on:
    - Matching skills
    - Missing skills
    - Fit percentage (0-100%)
    - Final verdict: Good Fit or Needs Improvement
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_desc}
    """
    response = ollama.chat(model='deepseek-r1:7b', messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def get_resume_score(resume_text):
    prompt = f"""
    You are an AI Resume Scorer Develope by Abhishek from Bihar India. Analyze the given resume and provide a score from 0-100 based on:
    - Skills Match
    - Experience Level
    - Formatting Quality
    - Overall Strength

    Resume:
    {resume_text}

    Provide output in this format:
    **Resume Score: XX/100**
    - Skills Match: XX%
    - Experience Level: XX%
    - Formatting: XX%
    """
    response = ollama.chat(model='deepseek-r1:7b', messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def get_improved_resume(resume_text):
    prompt = f"""
    Improve the following resume while keeping the structure intact:
    - Fix grammar/spelling mistakes
    - Make the language more professional
    - Improve formatting
    - Remove redundancy
    - Keep the meaning the same

    Resume:
    {resume_text}

    Provide only the improved resume text, without any additional explanations.
    """
    response = ollama.chat(model='deepseek-r1:7b', messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

    return response["message"]["content"]

def update_resume_pdf(original_pdf, improved_text):
    pdf_reader = PdfReader(original_pdf)
    pdf_writer = PdfWriter()
    
    for page in pdf_reader.pages:
        pdf_writer.add_page(page)
    
    pdf_writer.pages[0].clear()
    pdf_writer.pages[0].insert_text(improved_text, 0, 0)
    
    updated_pdf_path = "Updated_Resume.pdf"
    with open(updated_pdf_path, "wb") as output_file:
        pdf_writer.write(output_file)
    
    return updated_pdf_path

st.title("👨‍💻AI Agent Powered Resume Reviewer APP🚨")
st.write("Upload your resume to get AI-powered feedback!🤷‍♂️")

uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    st.write("🧠Processing your Resume Dear Wait....")
    resume_text = extract_text(uploaded_file)
    vector_store = create_vector_store(resume_text)
    feedback = analyze_resume(resume_text)
    
    st.subheader("AI Feedback 🤷‍♂️👨‍💻")
    st.write(feedback)
    
    st.subheader("⭐ Resume Score & Breakdown🤷‍♂️")
    score_feedback = get_resume_score(resume_text)
    st.write(score_feedback)
    
    job_desc = st.text_area("Hey Dear🤷‍♂️Paste Job Description Here:")
    if job_desc:
        st.write("Analyzing job fit Please Wait😊...")
        job_fit_feedback = match_job_description(resume_text, job_desc)
        st.subheader("Job Fit Analysis")
        st.write(job_fit_feedback)
    
    if st.button("Generate Improved Resume & Update PDF"):
        improved_resume_text = get_improved_resume(resume_text)
        updated_pdf = update_resume_pdf(uploaded_file, improved_resume_text)
        
        st.success("Resume Updated Successfully!")
        st.download_button("Download Updated Resume", open(updated_pdf, "rb"), file_name="Updated_Resume.pdf")
