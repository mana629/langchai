from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import TypedDict,Annotated,Optional
from dotenv import load_dotenv

load_dotenv()
model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.7,
        max_new_tokens=100,
)))

class review (TypedDict):
    key_themes : Annotated[list[str], "The key themes of the review"]
    summary : Annotated[str, "A brief summary of the review"]
    sentiment : Annotated[str, "The sentiment of the review (positive/negative/neutral)"]
    pros : Annotated[Optional[list[str]], "The positive aspects of the movie, if any"]
    cons : Annotated[Optional[list[str]], "The negative aspects of the movie, if any"]

structured_model = model.with_structured_output(review)

result = structured_model.invoke("""
The movie was fantastic! I loved the storyline and the acting was superb. The cinematography was breathtaking, and the soundtrack perfectly complemented the scenes. Overall, it was an amazing experience that I would highly recommend to others.

""")

print(result["summary"])
print(result["sentiment"])
print(result["key_themes"])
print(result["pros"])
print(result["cons"])