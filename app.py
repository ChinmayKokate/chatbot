import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📄 Chat with PDF 🤖 (Powered by Groq)")

# ====================== GROQ API SETUP ======================
if "client" not in st.session_state:
    if "GROQ_API_KEY" not in st.secrets:
        st.error("🚨 Missing Groq API Key!\n\nAdd `GROQ_API_KEY` in your Streamlit secrets.")
        st.stop()
    
    st.session_state.client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )

# ====================== PDF STORAGE ======================
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# ====================== UPLOAD PDF ======================
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        st.session_state.pdf_text = text
    st.success(f"✅ PDF loaded successfully! ({len(text.split())} words)")

# ====================== CHAT HISTORY ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ====================== CHAT INPUT ======================
user_input = st.chat_input("Ask anything from the uploaded PDF...")

if user_input:
    if not st.session_state.pdf_text.strip():
        st.warning("Please upload a PDF first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # System prompt with PDF content
        system_prompt = f"""You are an intelligent assistant. Answer the question **only using the information provided in the PDF below**. 
If the answer is not present in the document, clearly say "Sorry, I couldn't find this information in the uploaded PDF."

PDF Content:
{st.session_state.pdf_text}"""

        # Prepare messages for API
        full_messages = [
            {"role": "system", "content": system_prompt}
        ] + st.session_state.messages

        # Call Groq API
        with st.spinner("Thinking..."):
            response = st.session_state.client.chat.completions.create(
                model="llama-3.3-70b-versatile",     # Best balance of speed, quality & context
                messages=full_messages,
                temperature=0.2,
                max_tokens=1024,
                top_p=0.9,
            )
            answer = response.choices[0].message.content

        # Add assistant reply
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        st.rerun()
