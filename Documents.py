from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
from uuid import uuid4
load_dotenv()


llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-70b-versatile"
    )


model_name = "sentence-transformers/all-mpnet-base-v2"
embedding = HuggingFaceEmbeddings(model_name=model_name)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

loader = TextLoader("facts.txt")

docs = loader.load_and_split(
    text_splitter=text_splitter
)

retriver = Chroma.from_documents(docs, embedding=embedding).as_retriever(
    search_kwargs={"k": 6}
)

template = """You are a helpful, repectful and and honest assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the user's question. 
            If you don't know answer, just say you don't know, don't try to make up answer.

            {context}

            Question: {question}
        """

prompt = ChatPromptTemplate.from_template(template=template)

chain = (
    {"context": retriver, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type="stuff",
    retriever=retriver,
    chain_type_kwargs={"prompt": prompt},
    verbose=True
)



response = rag_chain.invoke("what are the intresting fact about english language")
print(response)



