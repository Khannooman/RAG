"""
VectorDB
"""

import shutil
import os
from typing import Literal, Sequence

from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores import Chroma, FAISS

from src import CFG, logger
from src.parser import load_pdf, simple_text_split, propositionize


def build_vectordb(filename: str, embedding_function: Embeddings) -> None:
    """Builds a vector database from a PDF file."""
    parts = load_pdf(filename)
    vectordb_path = CFG.VECTORDB[0].PATH

    if CFG.TEXT_SPLIT_MODE == "default":
        docs = simple_text_split(parts, chunk_size=500, chunk_overlap=100)
        save_vectordb(docs, embedding_function, vectordb_path)
    else:
        raise NotImplementedError
    

def save_vectordb(
        docs: Sequence[Document],
        embedding_function: Embeddings,
        persist_directory: str
) -> None:
    
    """Saves a vector database to disk"""
    logger.info(f"Save vectordb to {persist_directory}")

    vectorstore = Chroma(
        collection_name="langchain",
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    _ = vectorstore.add_documents(docs)


def load_vectordb(
        embedding_function: Embeddings,
        persist_directory: str
) -> VectorStore:
    
    """Loads a chroma index from dist"""
    logger.info(f"Load from {persist_directory}")

    vectorstore = Chroma(
        collection_name="langchain",
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    return vectorstore


def delete_vectordb(persist_directory) -> None:
    """Delete Vector database"""
    vectorstore = Chroma(
        collection_name='langchain',
        persist_directory=persist_directory,
    )
    vectorstore.delete_collection()

