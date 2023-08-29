import os
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
from ingestion.ingest import (load_single_document,load_document_batch,load_documents,split_documents)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from modules.analysis import analyze_and_answer
from modules.open_ai_embedding import get_openAi_embeddings
import click
import torch
from constants import (
    CHROMA_SETTINGS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


app = Flask(__name__)

UPLOAD_FOLDER = 'SOURCE_DOCUMENTS'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'md','ppt','pptx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'application/json'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"message": "Your file is uploaded successfully"}), 201
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question not provided"}), 400
    
    # documents = load_documents(SOURCE_DIRECTORY)
    # text_documents, python_documents = split_documents(documents)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # python_splitter = RecursiveCharacterTextSplitter.from_language(
    #     language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    # )
    # texts = text_splitter.split_documents(text_documents)
    # texts.extend(python_splitter.split_documents(python_documents))
    # embedding=get_openAi_embeddings()
    # vectordb = Chroma.from_documents(texts, embedding)
    # Sample text data for embedding (you need to modify this)
   
    # Get the result from analyze_and_answer
    result = analyze_and_answer(question, vectordb)
    print(result)
    
    return jsonify(result)

if __name__ == '__main__':
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    embedding=get_openAi_embeddings()
    vectordb = Chroma.from_documents(texts, embedding)
    print(vectordb)

    app.run(host='127.0.0.2', port=5000)
    
