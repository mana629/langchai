from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_huggingface import  HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "SACHIN TENDULKAR IS ALSO KNOWN AS THE GOD OF CRICKET."
"ROHIT SHARMA IS FAMOUSLY CALLED THE HITMAN."
"VIRAT KOHLI IS CELEBRATED AS THE KING OF CRICKET."
"JASPRIT BUMRAH, INDIA'S FAST BOWLER, IS NICKNAMED THE BULLET TRAIN."
"MS DHONI IS REMEMBERED AS CAPTAIN COOL FOR HIS CALM LEADERSHIP."
"YUVRAJ SINGH IS KNOWN AS THE SIXER KING AFTER HITTING SIX SIXES IN AN OVER."
"SOURAV GANGULY IS CALLED THE PRINCE OF KOLKATA."
"AB DE VILLIERS IS ADMIRED WORLDWIDE AS MR. 360 FOR HIS VERSATILE BATTING."
"CHRIS GAYLE IS KNOWN AS THE UNIVERSE BOSS FOR HIS EXPLOSIVE BATTING."
"RAHUL DRAVID IS RESPECTED AS THE WALL FOR HIS SOLID DEFENSE."

]
query = "ms  dhoni who?"
query_vector = embeddings.embed_query(query)
document_vectors = embeddings.embed_documents(documents)
similarities = cosine_similarity([query_vector], document_vectors)
print( similarities)