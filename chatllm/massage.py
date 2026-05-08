from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()
model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.5,
        max_new_tokens=100,
)))

massages= [

    SystemMessage(content="You are a helpful assistant."),
     
]
result = model.invoke(massages)

massages.append(AIMessage(content=result.content))
print(massages)