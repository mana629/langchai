from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

load_dotenv()
model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.5,
        max_new_tokens=100,
)))

templet1 = PromptTemplate(
    template = "write a dailtaied report on the following text : {text}",
    input_variables=["text"]
)
template2 = PromptTemplate(
    template = "write a 5 line summary of the following text : {text}",
    input_variables=["text"]
)
parser = StrOutputParser()

chain = templet1 | model | parser | template2 | model | parser
result = chain.invoke({
    "text" : "black hole"
})
print(result)