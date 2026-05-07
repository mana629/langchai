from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv 
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B"
    ,task="text-generation",))

Prompt= PromptTemplate(
    input_variables=["topic"],
    template="suggest a catchy blog title about {topic}."
)
topic = input("Enter a topic for the blog title: ")
result_prompt = Prompt.format(topic=topic)
blog_title = model.invoke(result_prompt)
print("Suggested Blog Title:", blog_title)