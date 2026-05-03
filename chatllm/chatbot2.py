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

chat_history = [

    SystemMessage(content="You are a helpful assistant."),
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))  # Add user input to chat history
    if user_input == "exit":
        print("Exiting chat...")
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print("ai :", result.content)

print(chat_history)