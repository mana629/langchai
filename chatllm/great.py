from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.7,
        max_new_tokens=100,
)))

class review (TypedDict):
    summary :str
    sentiment :str

structured_model = model.with_structured_output(review)

result = structured_model.invoke("""
The movie was fantastic! I loved the storyline and the acting was superb. The cinematography was breathtaking, and the soundtrack perfectly complemented the scenes. Overall, it was an amazing experience that I would highly recommend to others.

""")

print(result)