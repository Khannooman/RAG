"""
Retrival QA
"""

from typing import List

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter
)
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableBranch
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.load import dumps, loads
from src.prompt_template import QA_TEMPLATE, CONDENSE_QUESTION_TEMPLATE, MULTI_QUERY_TEMPLATE, CONTEXT_QUERY_TEMPLATE
import yaml
from src import CFG
from operator import itemgetter

class VectorStoreRetrieverWithScores(VectorStoreRetriever):
    def get_relavant_documents(self, query: str) -> List[Document]:
        if self.search_type == "similarity":
            docs_and_scores = self.vectorstore.similarity_search(
                query, **self.search_kwargs
            )
            for doc, score in docs_and_scores:
                doc.metadata = {**doc.metadata, "similarity_score": f"{score:.4f}"}
            docs = [doc for doc, _ in docs_and_scores]

        elif self.search_type == "similarity_score_thresold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )

            for doc , score in docs_and_similarities:
                doc.metadata = {**doc.metadata, "similarity_score": f"{score:.4f}"}
            docs = [doc for doc, _ in docs_and_similarities]

        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )

        else:
            raise ValueError(f"search_type of {self.search_type} not allowed")
        
        return docs


#### base retriver    
def build_base_retriever(vectordb: VectorStore) -> VectorStore:
    return VectorStoreRetrieverWithScores(
        vectorstore = vectordb, search_kwargs=({"k": CFG.BASE_RETRIEVER_CONFIG.SEARCH_K})
    )


def build_multivector_retriver(
        vectorstore: VectorStore, docstore
) -> VectorStoreRetriever:
    from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        search_type=SearchType.mmr
    )

def build_rerank_retriever(
        vectordb: VectorStore, reranker: BaseDocumentCompressor
) -> ContextualCompressionRetriever:
    base_retriever = VectorStoreRetrieverWithScores(
       vectorstore=vectordb, search_kwargs= {"k": CFG.BASE_RETRIEVER_CONFIG.SEARCH_K}
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )


def build_compression_retriever(
        vectordb: VectorStore, embeddings: Embeddings
) -> ContextualCompressionRetriever:
    base_retriever = VectorStoreRetrieverWithScores(
        vectorstore=vectordb,
        search_kwargs={"k": CFG["BASE_RETRIEVER_CONFIG"]["SEARCH_K"]}
    )

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relavant_filter = EmbeddingsFilter(
        embeddings=embeddings,
        similarity_threshold=CFG.COMPRESSION_RETRIVER_CONFIG.SIMILARITY_THRESOLD
    )
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relavant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever, base_compressor=pipeline_compressor
    )
    return compression_retriever


def condense_question_chain(llm: LLM):
    """Builds a chain that condense question and chat history to create a standalone question"""
    condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    chain = RunnableBranch(
        (
            #Both empty string and empty list evluate to False
            lambda x: not x.get("chat_history", False),
            # if no chat history, then we just pass input
            (lambda x: x["question"]),
        ),
        condense_question_prompt | llm | StrOutputParser
    )

    return chain



def build_question_answer_chain(llm: LLM) -> Runnable:
    """Builds a question answering chain"""

    qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)

    def format_docs(inputs: dict) -> str:
        return "\n\n".join(doc.page_content for doc in inputs["context"])
    
    question_answer_chain = (
        RunnablePassthrough.assign(context=format_docs).with_config(
            run_name="format_inputs"
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="stuff_documents_chain")
    return question_answer_chain




def build_rag_chain(llm: LLM, retriever: BaseRetriever) -> Runnable:
    """Builds a retrival RAG chain"""

    from langchain.chains.retrieval_qa.base import RetrievalQA

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
        verbose=True
    )

    return rag_chain


def build_conv_rag_chain(
        vectordb: VectorStore , llm: LLM
) -> Runnable:
    "Builds a Conversational RAG CHAIN"

    retriever = build_base_retriever(vectordb)

    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(QA_TEMPLATE)},
        condense_question_prompt=PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE),
        verbose=True
    )

    return rag_chain



def multiqueryRag(Vectordb: VectorStore, llm: LLM) -> Runnable:

    prompt = PromptTemplate.from_template(MULTI_QUERY_TEMPLATE)
    question_answer_prompt = PromptTemplate.from_template(CONTEXT_QUERY_TEMPLATE)
    retriever = Vectordb.as_retriever() 
    generate_queries = (
        prompt
        | llm
        | StrOutputParser()
        |(lambda x:x.split("\n"))
    )
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    final_rag_chain = (
        {'context': retrieval_chain,
         'question': itemgetter('question'),
         'chat_history': itemgetter('chat_history')
         }
        | question_answer_prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain


def get_unique_union(documents: list[list]) -> list:
    """Unique union of relative docs"""

    # Flatten list of lists, and convert each Document to string
    flatten_docs = [dumps(doc) for sublist in documents for doc in sublist]

    # Get unique documents
    unique_docs = list(set(flatten_docs))

    return [loads(doc) for doc in unique_docs]




 
    

