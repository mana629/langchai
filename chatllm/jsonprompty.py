from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.1,
        max_new_tokens=200,
)))

parser = StrOutputParser()

prompt = """Generate a detailed report on black holes."""

result = model.invoke(prompt)
final_result = parser.parse(result.content)

print(final_result)
print(type(final_result))
 