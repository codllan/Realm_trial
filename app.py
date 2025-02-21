##run command##  streamlit run c:/Users/Greenlight.Remote1/Documents/Programming.Files/ProgrammingProjects/realm-trial/Realm_trial/app.py [ARGUMENTS] --server.fileWatcherType=none
import streamlit as st
import pymongo
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64
import numpy as np
from datetime import datetime
import requests

# MongoDB connection
@st.cache_resource
def init_connection():
    connection_string = st.secrets["mongo"]["connection_string"]
    client = pymongo.MongoClient(connection_string)
    return client

client = init_connection()
db = client["vector_db"]
collection = db["data_entries"]

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# xAI API setup  "https://api.x.ai/v1/chat/completions"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"  # Replace with actual xAI URL
XAI_API_KEY = st.secrets["xai"]["api_key"]

# Function to chunk text
def chunk_text(text, max_length=200):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Function to vectorize text
def vectorize_text(text):
    return model.encode(text).tolist()

# Function to encode image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to query xAI
def detect_conflicts(chunks):
    embeddings = [vectorize_text(chunk["text"]) for chunk in chunks]
    conflicts = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if similarity < 0.3:
                conflicts.append((chunks[i]["text"], chunks[j]["text"]))
    return conflicts

def ask_xai(question, context_chunks):
    context = "\n".join([chunk["text"] for chunk in context_chunks])
    prompt = f"Based on the following data:\n\n{context}\n\nAnswer this question: {question}. Carefully consider all provided information. If the data contains conflicting statements, explain both perspectives, including any conditions or context (e.g., specific actions or settings) that affect the outcome."
    headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "grok-beta", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.7}
    try:
        response = requests.post(XAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"xAI API Error: {e}")
        return "Sorry, I couldn’t process your question."

# Streamlit UI
st.title("Vector Data Collector with xAI Question-Answering")

# Tabs
tab1, tab2 = st.tabs(["Enter Data", "Ask a Question"])

# --- Data Entry Tab ---
with tab1:
    st.header("Enter Your Data")
    main_text = st.text_area("Main Paragraph", height=200)
    st.subheader("Add Screenshots and Notes")
    num_slides = st.number_input("Number of Slides", min_value=1, max_value=5, value=1)
    slides = []
    for i in range(int(num_slides)):
        st.write(f"Slide {i+1}")
        screenshot = st.file_uploader(f"Screenshot {i+1}", type=["png", "jpg", "jpeg"], key=f"screen_{i}")
        notes = st.text_input(f"Notes for Slide {i+1}", key=f"notes_{i}")
        if screenshot:
            image = Image.open(screenshot)
            slides.append({"screenshot": image_to_base64(image), "notes": notes})
    
    if st.button("Upload"):
        if main_text.strip():
            text_chunks = chunk_text(main_text)
            chunk_embeddings = [vectorize_text(chunk) for chunk in text_chunks]
            for chunk, embedding in zip(text_chunks, chunk_embeddings):
                doc = {
                    "original_text": main_text,
                    "chunk_text": chunk,
                    "embedding": embedding,
                    "slides": slides,
                    "timestamp": datetime.utcnow()
                }
                collection.insert_one(doc)
            st.success("Data uploaded successfully!")
        else:
            st.error("Please enter some text.")

# --- Question-Answering Tab ---
with tab2:
    st.header("Ask a Question")
    question = st.text_input("Enter your question about the data")
    if st.button("Ask"):
        if question.strip():
            query_embedding = vectorize_text(question)
            st.write("Debug: Query Embedding Sample:", query_embedding[:5])
            pipeline = [
                {"$vectorSearch": {"index": "vector_index", "path": "embedding", "queryVector": query_embedding, "limit": 10, "numCandidates": 1000}},
                {"$project": {"original_text": 1, "chunk_text": 1, "slides": 1, "score": {"$meta": "vectorSearchScore"}}},
                {"$limit": 3}
            ]
            try:
                results = list(collection.aggregate(pipeline))
                st.write("Debug: Raw Vector Search Results:", results)
                if results:
                    context_chunks = [{"text": r["chunk_text"], "score": r["score"]} for r in results]
                    conflicts = detect_conflicts(context_chunks)
                    st.write("Debug: Context Chunks:", context_chunks)
                    if conflicts:
                        st.warning("Conflicts detected:")
                        for c1, c2 in conflicts:
                            st.write(f"- '{c1}' vs '{c2}'")
                    answer = ask_xai(question, context_chunks)
                    st.write("**Answer:**", answer)
                    st.write("**Referenced Chunks and Images:**")
                    for i, result in enumerate(results, 1):
                        st.write(f"{i}. {result['chunk_text']} (Score: {result['score']})")
                        if "slides" in result and result["slides"]:
                            for slide in result["slides"]:
                                image_data = base64.b64decode(slide["screenshot"])
                                st.image(image_data, caption=slide["notes"], width=300)
                else:
                    st.write("No relevant data found to answer your question.")
            except pymongo.errors.OperationFailure as e:
                st.error(f"MongoDB Error: {e}")
        else:
            st.error("Please enter a question.")

# Instructions
st.sidebar.write("Run with: `streamlit run app.py --server.fileWatcherType=none`")
st.sidebar.write("Ensure MongoDB Atlas vector_index and xAI API key are set up.")