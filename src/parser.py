"""
Parser
"""

import uuid
from typing import Sequence

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean
from src import CFG
# import config as CFG

def load_pdf(filename: str) -> Sequence:
    """Loads pdf."""
    # from pdfminer import psparser
    # return partition_pdf(filename, strategy="fast")
    from langchain_community.document_loaders import PyMuPDFLoader
    return PyMuPDFLoader(filename).load()


def simple_text_split(
        docs:Sequence[Document], chunk_size: int, chunk_overlap: int
) -> Sequence[Document]:
    """Text chunking using langchain RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    texts = text_splitter.split_documents(docs)

    for t in texts:
        t.metadata["page_number"] = t.metadata["page"] + 1
    return texts


def parent_document_split(
        docs: Sequence[Document],
) -> tuple[Sequence[Document], tuple[list[str], Sequence[Document]]]:
    """Text chunking for ParentDocumentRetriever"""
    id_key = "doc_id"

    parent_docs = simple_text_split(docs, 200, 0)
    doc_ids = [str(uuid.uuid4()) for _ in parent_docs]

    child_docs = []
    for i, pdoc in enumerate(parent_docs):
        _sub_docs = simple_text_split([pdoc], 50, 0)
        for _doc in _sub_docs:
            _doc.metadata[id_key] = doc_ids[i]
            child_docs.extend(_doc)
    return child_docs, (doc_ids, parent_docs)

 
def propositionize(docs: Sequence[Document]) -> Sequence[Document]:
    """Text chunking with Propositioner"""
    from src.elements.propositioner import Propositioner
    propositioner = Propositioner()

    text = simple_text_split(
        docs,
        CFG.PROPOSITIONER_CONFIG.CHUNK_SIZE,
        CFG.PROPOSITIONER_CONFIG.CHUNK_OVERLAP,

    )

    prop_text = propositioner.batch(text)
    return prop_text
