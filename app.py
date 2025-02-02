import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate

# Set up Streamlit page
st.set_page_config(page_title="Chat With Multiple PDFs", layout="wide")

# Define FAISS storage path
FAISS_PATH = "faiss_index"

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")  # Ensure API Key is set in Streamlit Secrets
if not api_key:
    st.error("Google API key is missing! Please set it in Streamlit secrets.")
else:
    genai.configure(api_key=api_key)


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Avoid NoneType errors
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Ensure directory exists before saving
    if not os.path.exists(FAISS_PATH):
        os.makedirs(FAISS_PATH, exist_ok=True)

    vector_store.save_local(FAISS_PATH)
    st.session_state["vector_store"] = FAISS_PATH  # Store the FAISS path in session


# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the questions as detailed as possible from the provided context.
    If the answer is not in the provided context, say: "Answer is not available in the given file."
    
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Function to process user input
def user_input(user_question):
    if "vector_store" not in st.session_state or not os.path.exists(st.session_state["vector_store"]):
        st.error("Please upload and process PDFs first!")
        return

    try:
        # Load FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(st.session_state["vector_store"], embeddings)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("Reply: ", response.get("output_text", "No response generated."))
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return


# Main function for Streamlit UI
def main():
    st.header("Chat with Multiple PDFs powered by Gemini 🚀")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)

                st.session_state["pdf_docs"] = pdf_docs
                st.success("PDFs processed successfully!")

    if "pdf_docs" in st.session_state:
        st.subheader("Uploaded Files:")
        for i, pdf_doc in enumerate(st.session_state["pdf_docs"]):
            st.write(f"{i + 1}. {pdf_doc.name}")

    if st.button("Reset"):
        if os.path.exists(FAISS_PATH):
            import shutil
            shutil.rmtree(FAISS_PATH)  # Delete FAISS index
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()
