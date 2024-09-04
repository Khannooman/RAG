from langchain_core.prompts import ChatPromptTemplate 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()  
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.1-70b-versatile"
    )

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)
test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
    
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"]   
)

result = code_chain({
    "language": args.language,
    "task": args.task
})

print("code----", result["code"])
print("test----", result["test"])