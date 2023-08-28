import os
import sys
from pathlib import Path
#for appending the file address
sys.path.append(str(Path(__file__).parent.parent))
from logger import logging
from langchain.vectorstores import Chroma
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from modules.open_ai_embedding import get_openAi_embeddings
#from modules.image_processing import extract_image_features
#from modules.vector_extraction import extract_vectors_from_pdf
from modules.data_fusion import perform_data_fusion
from modules.analysis import analyze_and_answer
from ingestion.ingest import (load_single_document,load_document_batch,load_documents,split_documents)
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import click
import torch
from chromadb.config import Settings

from constants import (
    SOURCE_DIRECTORY,
)
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, "DB")
# Define the updated Chroma settings
CHROMA_SETTINGS = {
    "database": {
        "implementation": "duckdb+parquet",
        "persist_directory": PERSIST_DIRECTORY,
        "anonymized_telemetry": False
    }
}


os.environ["OPENAI_API_KEY"] = "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"
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(device_type,show_sources):
    #step 1. data ingestion and loading
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    # Create OpenAIEmbeddings and Chroma
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    # print(texts)
    texts.extend(python_splitter.split_documents(python_documents))
    embedding=get_openAi_embeddings()
    db = Chroma.from_documents(texts, embedding,client_settings=CHROMA_SETTINGS,persist_directory=PERSIST_DIRECTORY)
    db.persist()
    db=None
   
   
if __name__=="__main__":
    main()