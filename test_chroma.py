import chromadb

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    "video_rag"
)

print("Total Chunks :", collection.count())