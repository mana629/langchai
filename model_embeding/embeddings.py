from langchain_huggingface import  HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = "asansol is a city in west bengal."
vector = embeddings.embed_query(text)
print(str(vector))