from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from dotenv import load_dotenv
load_dotenv()
from Documents import llm, retriver

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriver
)

compressed_docs = compression_retriever.invoke(
    "what is the intresting fact about english"
)

print(compressed_docs)
