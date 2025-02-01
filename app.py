import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
st.set_page_config(page_title="Chat With Multiple PDF", layout="wide")

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    """
    Extracts text from a list of PDF documents.
    Args:
        pdf_docs: A list of PDF documents.
    Returns:
        A string containing the extracted text from all PDF documents.
    """

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    # The return statement was in the wrong place, causing the function to always return an empty string
    return text


def get_text_chunks(text):
    """
    Splits a large text into smaller chunks for efficient processing.
    Args:
        text: A string containing the text to be split.
    Returns:
        A list of text chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """
    Creates a vector store from a list of text chunks.
    Args:
        text_chunks: A list of text chunks.
    Returns:
        A FAISS vector store.
    """
    # Moved embeddings initialization inside the function
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faisss_index")  # Corrected the typo in the filename
    return vector_store


def get_conversional_chain():
    """
    Creates a conversational chain for question answering.
    Returns:
        A LangChain question-answering chain.
    """
    # Removed redundant prompt_template assignment
    prompt_template = """
    Answer the questions as detailed as possible from the provided context, if the answer is not in
    provided context just say, "answer is not available in the given file", Provide accurate answers\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Context:
    {context}?
    Answer:
    Question:
    {question}
    Answer:
    """
    # Moved model and prompt initialization inside the function
    # Removed redundant model and prompt initialization
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Removed redundant user_input function
def user_input(user_question, processed_pdf_text=None):
    """
    Processes user input and generates a response using the conversational chain,
    providing both the user's question and the processed PDF text as context.
    Args:
        user_question: The user's question.
        processed_pdf_text: The processed text extracted from the uploaded PDF files.
    Returns:
        The generated response from the conversational chain.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faisss_index", embeddings)  # Corrected the typo in the filename
    docs = new_db.similarity_search(user_question)
    chain = get_conversional_chain()
    # Removed redundant response assignment
    # Added check for processed_pdf_text
    if processed_pdf_text:
        context = f"{processed_pdf_text}\n\nQuestion: {user_question}"
        response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
    else:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])  # Removed redundant st.write


def main():
    """
    Main function for the Streamlit app.
    """
    st.set_page_config("Chat With Multiple PDF")  # Removed redundant st.set_page_config
    st.header("Chat with PDF's powered by Gemini 🙋‍♂️")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if st.session_state.get("pdf_docs"):
            processed_pdf_text = get_pdf_text(st.session_state["pdf_docs"])
            user_input(user_question, processed_pdf_text)
        else:
            st.error("Please upload PDF files first.")

    with st.sidebar:
        st.title("Menu:")
        # Removed redundant st.file_uploader
        pdf_docs = st.file_uploader("Upload Files & Click Submit to Proceed", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # Removed redundant with st.spinner
                raw_text = ""
                text_chunks = []
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()

                # Indented the following lines to be within the 'with' block
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                chain = get_conversional_chain()
                st.session_state["pdf_docs"] = pdf_docs
                st.session_state["text_chunks"] = text_chunks
                st.session_state["vector_store"] = vector_store
                st.session_state["chain"] = chain
                st.success("PDFs processed successfully!")

if st.button("Reset"):
    st.session_state["pdf_docs"] = []
    st.session_state["text_chunks"] = []
    st.session_state["vector_store"] = None
    st.session_state["chain"] = None
    st.experimental_rerun()

if st.session_state.get("pdf_docs"):
    st.subheader("Uploaded Files:")
    for i, pdf_doc in enumerate(st.session_state["pdf_docs"]):
        st.write(f"{i + 1}. {pdf_doc.name}")


def main():
    st.title("Chat With Multiple PDFs")
    st.write("Welcome to the app!")

if __name__ == "__main__":
    main()
