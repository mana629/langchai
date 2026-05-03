from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
 

load_dotenv()

model = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.7,  # Increased for more variation
        max_new_tokens=100,
))) 
chat_history = []
while True:
    user_input = input("You: ")
    chat_history.append(user_input)  # Add user input to chat history
    if user_input.lower() == "exit":
        print("Exiting chat...")
        break
    # Use HumanMessage for proper chat format
    
    result = model.invoke(chat_history + [user_input])
    chat_history.append(result.content)

    print(f'Bot: {result.content}')