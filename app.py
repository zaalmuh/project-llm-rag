import os
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from litellm import completion
from langchain_community.tools import ArxivQueryRun
from dotenv import load_dotenv
from huggingface_hub import login
from utils import extract_text_from_pdfs, process_text_and_store, semantic_search
from models import generate_response
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if huggingface_token:
    login(token=huggingface_token)

arxiv_tool = ArxivQueryRun()


def main():
    st.title("RAG-powered Research Paper Assistant")

    # Option to choose between PDF upload and arXiv search
    option = st.radio("Choose an option:", ("Upload PDFs", "Search arXiv"))
    provider = st.selectbox(
        "Choose a provider:",
        ("huggingface", "gemini", "groq"), 
        index=0
    )
    if option == "Upload PDFs":
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            st.write("Processing uploaded files...")
            all_text = extract_text_from_pdfs(uploaded_files)
            collection = process_text_and_store(all_text)
            st.success("PDF content processed and stored successfully!")

            query = st.text_input("Enter your query:")
            if st.button("Execute Query") and query:
                results = semantic_search(query, collection)
                context = "\n".join(results['documents'][0])
                response = generate_response(query, context, provider=provider)
                st.subheader("Generated Response:")
                st.write(response)

    elif option == "Search arXiv":
        query = st.text_input("Enter your search query for arXiv:")

        if st.button("Search ArXiv") and query:
            arxiv_results = arxiv_tool.invoke(query)
            st.session_state["arxiv_results"] = arxiv_results  
            st.subheader("Search Results:")
            st.write(arxiv_results)

            collection = process_text_and_store(arxiv_results)
            st.session_state["collection"] = collection  

            st.success("arXiv paper content processed and stored successfully!")

        # Only allow querying if search has been performed
        if "arxiv_results" in st.session_state and "collection" in st.session_state:
            query = st.text_input("Ask a question about the paper:")
            if st.button("Execute Query on Paper") and query:
                results = semantic_search(query, st.session_state["collection"])
                context = "\n".join(results['documents'][0])
                response = generate_response(query, context, provider=provider)
                st.subheader("Generated Response:")
                st.write(response)

if __name__ == "__main__":
    main()