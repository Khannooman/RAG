from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.base import RunnableSequence

from prompt_template import MULTI_QUERY_TEMPLATE, GENRATED_RESULT_TEMPLATE


def build_llm_chain(llm: LLM, template: str) -> RunnablePassthrough:
    prompt = PromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    return chain

def build_multiple_queries_expansion_chain(llm: LLM) -> RunnablePassthrough:
    chain = {"question": RunnablePassthrough()} | build_llm_chain(
        llm, MULTI_QUERY_TEMPLATE
    )
    return chain 

def build_generated_result_expansion_chain(llm: LLM) -> RunnablePassthrough:
    chain = {"question": RunnablePassthrough()} | build_llm_chain(
        llm, GENRATED_RESULT_TEMPLATE
    )

    return chain