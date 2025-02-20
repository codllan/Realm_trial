import streamlit as st
import pymongo
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64
import numpy as np
from datetime import datetime

# MongoDB connection
@st.cache_resource
def init_connection():
    connection_string = st.secrets["mongo"]["connection_string"]
    client = pymongo.MongoClient(connection_string)
    return client

client = init_connection()
db = client["vector_db"]  # Your database name
collection = db["data_entries"]  # Your collection name

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

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
    return model.encode(text).tolist()  # Convert to list for MongoDB storage

# Function to encode image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Streamlit UI
st.title("Vector Data Collector with Semantic Search")

# Tabs for data entry and search
tab1, tab2 = st.tabs(["Enter Data", "Semantic Search"])

# --- Data Entry Tab ---
with tab1:
    st.header("Enter Your Data")
    
    # Main paragraph text
    main_text = st.text_area("Main Paragraph", height=200)
    
    # Slides for screenshots and notes
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
    
    # Upload button
    if st.button("Upload"):
        if main_text.strip():
            # Chunk the main text
            text_chunks = chunk_text(main_text)
            chunk_embeddings = [vectorize_text(chunk) for chunk in text_chunks]
            
            # Prepare document for MongoDB
            doc = {
                "main_text": main_text,
                "chunks": [{"text": chunk, "embedding": embedding} for chunk, embedding in zip(text_chunks, chunk_embeddings)],
                "slides": slides,
                "timestamp": datetime.utcnow()
            }
            
            # Insert into MongoDB
            collection.insert_one(doc)
            st.success("Data uploaded successfully!")
        else:
            st.error("Please enter some text in the main paragraph.")

# --- Semantic Search Tab ---
with tab2:
    st.header("Semantic Search")
    query = st.text_input("Enter your search query")
    if st.button("Search"):
        if query.strip():
            # Vectorize the query
            query_embedding = vectorize_text(query)
            
            # Perform vector search (assumes a vector index is created in MongoDB Atlas)
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Name of your vector index in MongoDB
                        "path": "chunks.embedding",
                        "queryVector": query_embedding,
                        "limit": 3,
                        "numCandidates": 100
                    }
                },
                {"$project": {"main_text": 1, "chunks.text": 1, "score": {"$meta": "vectorSearchScore"}}}
            ]
            
            results = list(collection.aggregate(pipeline))
            
            if results:
                st.write("Top 3 Matching Chunks:")
                for i, result in enumerate(results, 1):
                    st.write(f"**Result {i} (Score: {result['score']:.4f})**")
                    for chunk in result["chunks"]:
                        st.write(f"- {chunk['text']}")
                    st.write(f"From: {result['main_text'][:200]}...")
                    st.write("---")
            else:
                st.write("No matches found.")
        else:
            st.error("Please enter a search query.")

# Instructions for running
st.sidebar.write("Run this app with: `streamlit run app.py`")
st.sidebar.write("Ensure MongoDB Atlas Vector Search index 'vector_index' is set up on 'chunks.embedding'.")