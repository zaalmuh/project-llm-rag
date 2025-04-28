import os
import PyPDF2
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from litellm import completion
from langchain_community.tools import ArxivQueryRun
from dotenv import load_dotenv
from huggingface_hub import login
from models import generate_response
from litellm import completion
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = chromadb.Client() # Use a client to connect to a running ChromaDB instance
text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

if huggingface_token:
    login(token=huggingface_token)

arxiv_tool = ArxivQueryRun()
def generate_response(query, context, provider):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    if provider == "huggingface":
        api_key = os.getenv("HUGGINGFACE_TOKEN")
        model = "huggingface/HuggingFaceH4/zephyr-7b-beta"  # HuggingFace Zephyr
        response = completion(
            model=model,
            messages=[{"content": prompt, "role": "user"}],
            api_key=api_key,
        )
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        model = "llama3-70b-8192"  # Groq Llama3 70B
        response = completion(
            model=model,
            messages=[{"content": prompt, "role": "user"}],
            api_key=api_key,
            api_base="https://api.groq.com/openai/v1",  # Groq needs this
        )
    else:
        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[{"content": prompt, "role": "user"}],
            api_key=gemini_api_key,
        )
    return response['choices'][0]['message']['content']

def extract_text_from_pdfs(uploaded_files):
    texts = ""
    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            texts += page.extract_text() or ""
    return texts
def semantic_search(query, collection, top_k=2):
    query_embedding = text_embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results

def process_text_and_store(texts):
    # client = chromadb.PersistentClient(path='chroma_db')  # Use a persistent ChromaDB instance so the data is saved to disk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, 
                                                   separators=["\n\n", "\n", " ", ""])
    chunks = text_splitter.split_text(texts)
    try:
        client.delete_collection(name="knowledge_base")
    except Exception:
        pass
    collection = client.create_collection(name="knowledge_base")

    for i, chunk in enumerate(chunks):
        embedding=text_embedding_model.encode(chunk)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding.tolist()],
            documents=[chunk],
            metadatas=[{"source":"pdf", "chunk_id": i}]
        )

    return collection

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