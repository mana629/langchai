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
 
parser = JsonOutputParser()
templete = PromptTemplate(
    template = "write a dailtaied report on the following text : {formation_instructions}",
    input_variables=[],
    partial_variables={"formation_instructions" :  parser.get_format_instructions()},

)
prompt = templete.format()

print(prompt)