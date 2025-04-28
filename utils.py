import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from litellm import completion
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(uploaded_files):
    texts = ""
    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            texts += page.extract_text() or ""
    return texts
def semantic_search(query, collection, top_k=2):
    text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = text_embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results

def process_text_and_store(texts):
    text_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # client = chromadb.PersistentClient(path='chroma_db')  # Use a persistent ChromaDB instance so the data is saved to disk
    client = chromadb.Client() # Use a client to connect to a running ChromaDB instance
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