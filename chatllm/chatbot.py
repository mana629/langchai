from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.7,  # Increased for more variation
        max_new_tokens=200,
))) 

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chat...")
        break
    # Use HumanMessage for proper chat format
    messages = [HumanMessage(content=user_input)]
    result = model.invoke(messages)
    print("Mana:", result.content)