"""from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# Build pipeline from Hugging Face Hub
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
          
    )
)

# Wrap in Chat interface
model = ChatHuggingFace(llm=llm)"""

# Ask a test question
   # .content gives clean text output

from langchain_huggingface import  HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = "asansol is a city in west bengal."
vector = embeddings.embed_query(text)
print(str(vector))
