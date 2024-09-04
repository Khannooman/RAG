"""
Embeddings
"""

import os

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder, LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


from src.prompt_template import HYDE_TEMPLATE
import yaml

from src import CFG

def build_base_embeddings():
    """Builds base embeddings define in config."""
    base_embeddings = HuggingFaceEmbeddings(
        model_name=os.path.join(CFG.EMBEDDING_MODEL),
        model_kwargs={"device": CFG.DEVICE}
    )
    return base_embeddings



def build_hyde_embeddings(llm, base_embeddings):
    """Builds hypothetical documents embeddings."""
    prompt = PromptTemplate.from_template(HYDE_TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )

    return embeddings
