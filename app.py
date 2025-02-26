##run command##  streamlit run c:/Users/Greenlight.Remote1/Documents/Programming.Files/ProgrammingProjects/realm-trial/Realm_trial/app.py [ARGUMENTS] --server.fileWatcherType=none
import streamlit as st
import pymongo
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64
from datetime import datetime
import requests
import uuid

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

# xAI API setup
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
XAI_API_KEY = st.secrets["xai"]["api_key"]

# Function to vectorize text
def vectorize_text(text):
    return model.encode(text).tolist()

# Function to encode image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to query xAI
def ask_xai(question, context_chunks):
    context = "\n".join([chunk["text"] for chunk in context_chunks])
    prompt = f"Based on the following data:\n\n{context}\n\nAnswer this question: {question}. Carefully consider all provided information."
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
st.title("Data Collection & xAI Query")

# Tabs
tab1, tab2 = st.tabs(["Enter Data", "Ask a Question"])

# --- Data Entry Tab ---
with tab1:
    st.header("")
    
    # Initialize session state for input fields and slides
    if "subject" not in st.session_state:
        st.session_state.subject = ""
    if "main_text" not in st.session_state:
        st.session_state.main_text = ""
    if "slide_count" not in st.session_state:
        st.session_state.slide_count = 1
    if "notes" not in st.session_state:
        st.session_state.notes = {}
    if "slides_draft" not in st.session_state:
        st.session_state.slides_draft = {}  # Store file data as base64

    # Subject and Main Paragraph inputs
    subject = st.text_input("Subject", value=st.session_state.subject, key="subject_input")
    main_text = st.text_area("Main Paragraph", value=st.session_state.main_text, height=200, key="main_text_input")
    st.subheader("Add Screenshots and Notes")

    # Store slide data
    slides = []
    for i in range(st.session_state.slide_count):
        st.write(f"Slide {i+1}")
        screenshot_key = f"screen_{i}"
        notes_key = f"notes_{i}"

        # File uploader with draft persistence
        screenshot = st.file_uploader(f"Screenshot {i+1}", type=["png", "jpg", "jpeg"], key=screenshot_key)
        if screenshot:
            # Convert uploaded file to base64 and store in session state
            image = Image.open(screenshot)
            st.session_state.slides_draft[screenshot_key] = {"screenshot": image_to_base64(image), "name": screenshot.name}
        
        # Display the persisted file name if available
        if screenshot_key in st.session_state.slides_draft:
            st.write(f"Uploaded: {st.session_state.slides_draft[screenshot_key]['name']}")
        
        # Notes field
        if notes_key not in st.session_state.notes:
            st.session_state.notes[notes_key] = ""
        notes = st.text_input(f"Notes for Slide {i+1}", value=st.session_state.notes[notes_key], key=notes_key)

        # Add slide to list if it has a file
        if screenshot_key in st.session_state.slides_draft:
            slides.append({"screenshot": st.session_state.slides_draft[screenshot_key]["screenshot"], "notes": notes})
            # Add a new box if this is the last one and has a file
            if i == st.session_state.slide_count - 1:
                st.session_state.slide_count += 1

    if st.button("Upload"):
        if subject.strip() and main_text.strip():
            doc_id = str(uuid.uuid4())
            chunks = [
                {"type": "subject", "text": subject, "slides": []},
                {"type": "main_text", "text": main_text, "slides": []}
            ]
            for slide in slides:
                slide_text = slide["notes"] if slide["notes"] else "Slide with no notes"
                chunks.append({"type": "slide", "text": slide_text, "slides": [slide]})

            for chunk in chunks:
                embedding = vectorize_text(chunk["text"])
                doc = {
                    "doc_id": doc_id,
                    "subject": subject,
                    "original_text": main_text,
                    "chunk_type": chunk["type"],
                    "chunk_text": chunk["text"],
                    "embedding": embedding,
                    "slides": chunk["slides"],
                    "timestamp": datetime.utcnow()
                }
                collection.insert_one(doc)
            st.success(f"Data uploaded successfully! Document ID: {doc_id}")
            
            # Clear all fields and reset draft
            st.session_state.subject = ""
            st.session_state.main_text = ""
            st.session_state.notes = {}
            st.session_state.slides_draft = {}
            st.session_state.slide_count = 1
        else:
            st.error("Please enter both a subject and some text.")

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
                {"$project": {"doc_id": 1, "subject": 1, "original_text": 1, "chunk_type": 1, "chunk_text": 1, "slides": 1, "score": {"$meta": "vectorSearchScore"}}},
                {"$limit": 6}
            ]
            try:
                results = list(collection.aggregate(pipeline))
                st.write("Debug: Raw Vector Search Results:", results)
                if results:
                    context_chunks = [{"text": r["chunk_text"], "score": r["score"]} for r in results]
                    st.write("Debug: Context Chunks:", context_chunks)
                    answer = ask_xai(question, context_chunks)
                    st.write("**Answer:**", answer)
                    st.write("**Referenced Chunks and Images:**")
                    for i, result in enumerate(results, 1):
                        doc_id = result.get("doc_id", "N/A")
                        st.write(f"{i}. [Doc ID: {doc_id}] [{result['chunk_type']}] Subject: {result['subject']} - {result['chunk_text']} (Score: {result['score']})")
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
# st.sidebar.write("Run with: `streamlit run app.py --server.fileWatcherType=none`")
# st.sidebar.write("Ensure MongoDB Atlas vector_index and xAI API key are set up.")