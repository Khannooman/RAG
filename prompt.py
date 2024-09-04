from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import ChatMessagePromptTemplate, MessagesPlaceholders
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings()
db = Chroma(
    persist_directory="emb", 
    embedding_function=embeddings
)

retriver = db.as_retriever()

# SYSTEM_TEMPLATE = """
# Answer the user's question based on the below context.
# If the context doesn't contain any relevant information about
# the question, don't make something up just say I don't know

# <context>
# {context}
# </context>
# """

# question_answering_prompt = ChatMessagePromptTemplate(
#     [
#         (
#             "system",
#             SYSTEM_TEMPLATE,
#         ),
#         MessagesPlaceholders(variable_names="message")
#     ]
# )

rag_chain = RetrievalQA.from_chain_type(
    llm
)
