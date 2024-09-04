"""
LLM
"""
import os

from langchain.callbacks import StreamingStdOutCallbackHandler
import yaml

from src import CFG

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_REPETITION_PENALITY = 0.2
DEFAULT_CONTEXT_LENGTH = 4000

def build_llm():
    """Builds LLM defined in config"""
    if CFG.LLM_TYPE == "groq":
        from dotenv import load_dotenv

        _ = load_dotenv()
        return chatgroq(
            CFG.LLM_PATH,
            config = {
                "temperature": CFG.LLM_CONFIG.TEMPERATURE
            },   
        )



def chatgroq(
        model_name: str = "mixtral-8x7b-32768", config: dict | None = None, **kwargs
):
    """For Groq LLM."""
    from langchain_groq import ChatGroq

    if config is None:
        config = {
            "max_tokens" : DEFAULT_MAX_NEW_TOKENS,
            "temperature" : DEFAULT_MAX_NEW_TOKENS
        }

    llm = ChatGroq(
        model_name=model_name,
        **config,
        streaming=True,
    )
    return llm

