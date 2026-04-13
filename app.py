import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("📄 Chat with PDF 🤖")

# Store PDF text
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    st.session_state.pdf_text = text
    st.success("PDF loaded successfully!")

# Function to answer questions
def get_answer(question, text):
    sentences = text.split(".")
    
    vectorizer = TfidfVectorizer().fit(sentences + [question])
    vectors = vectorizer.transform(sentences + [question])
    
    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    
    index = similarity.argmax()
    
    return sentences[index]

# Chat section
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("Ask a question from the PDF...")

if user_input and st.session_state.pdf_text:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    answer = get_answer(user_input, st.session_state.pdf_text)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    st.rerun()