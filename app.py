
from flask import Flask, request, render_template, jsonify
import os
import faiss
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

app = Flask(__name__)

# === PDF Processing ===
pdf_folder_path = 'datasets'
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

all_text = ""
for pdf_file in pdf_files:
    all_text += extract_text_from_pdf(pdf_file) + "\n\n"

# === Chunk Text ===
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

chunks = chunk_text(all_text)

# === Embedding and FAISS Index ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks, show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# === Google Gemini Setup ===
genai.configure(api_key="AIzaSyDu63svNYVxNnH_IH6GzmcoNMZGUe-YZ3s")
model = genai.GenerativeModel('gemini-2.0-flash')

# === Retrieval Function ===
def retrieve(query, k=3):
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb).astype('float32'), k)
    return [chunks[i] for i in I[0]]

conversation_history = []

def rag_answer(query):
    context = '\n'.join(retrieve(query))
    history_str = ""
    for turn in conversation_history:
        history_str += f"User: {turn['query']}\nAssistant: {turn['response']}\n"

    prompt = f"""Use the following context and conversation history to answer the question. Keep your answer concise and relevant.

Context:
{context}

Conversation History:
{history_str}
User: {query}
Assistant:"""

    response = model.generate_content(prompt)
    answer = response.text.strip()
    conversation_history.append({'query': query, 'response': answer})
    return answer

# === Flask Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('query')
    if user_input:
        response = rag_answer(user_input)
        return jsonify({'response': response})
    return jsonify({'response': 'No query received'}), 400

if __name__ == '__main__':
    app.run(debug=True)
