
# Import Required Libraries


import json
import pickle

from pathlib import Path

import chromadb
import ollama

from rank_bm25 import BM25Okapi


# Hybrid Retriever


class HybridRetriever:

    def __init__(self):

       
        # ChromaDB
        
        self.client = chromadb.PersistentClient(

            path="data/chroma_db"

        )

        self.collection = self.client.get_collection(

            "video_rag"

        )

        
        # BM25 Folder
        
        self.bm25_directory = Path(

            "data/bm25_index"

        )

        
        # Embedding Model
        

        self.embedding_model = "nomic-embed-text"

        print("\n========================================")
        print("Hybrid Retriever Ready")
        print("========================================")

# Load Embedding Model

from sentence_transformers import SentenceTransformer


class HybridRetriever:

    def __init__(self):

        
        # ChromaDB
        

        self.client = chromadb.PersistentClient(

            path="data/chroma_db"

        )

        self.collection = self.client.get_collection(

            "video_rag"

        )

        
        # BM25 Folder
       

        self.bm25_directory = Path(

            "data/bm25_index"

        )

        
        # Dense Embedding Model
        

        print("Loading SentenceTransformer...")

        self.embedding_model = SentenceTransformer(

            "all-MiniLM-L6-v2"

        )

        print("✓ Model Loaded")
   
    # Generate Query Embedding
    
    def generate_query_embedding(

        self,

        query

    ):

        embedding = self.embedding_model.encode(

            query,

            convert_to_numpy=True

        )

        return embedding.tolist()
    
    # Dense Search using ChromaDB
    

    def dense_search(

        self,

        query,

        top_k=5

    ):

        print("\nPerforming Dense Search...")

        
        # Generate Query Embedding
        

        query_embedding = self.generate_query_embedding(

            query

        )

        
        # Search ChromaDB
        

        results = self.collection.query(

            query_embeddings=[

                query_embedding

            ],

            n_results=top_k

        )

        dense_results = []

        
        # Store Results
        
        ids = results["ids"][0]

        documents = results["documents"][0]

        metadatas = results["metadatas"][0]

        distances = results["distances"][0]

        for i in range(

            len(ids)

        ):

            dense_results.append(

                {

                    "chunk_id": ids[i],

                    "text": documents[i],

                    "video_id": metadatas[i]["video_id"],

                    "type": metadatas[i]["type"],

                    "timestamp": metadatas[i]["timestamp"],

                    "score": distances[i]

                }

            )

        print(

            f" Retrieved {len(dense_results)} Dense Results"

        )

        return dense_results
    
    # Sparse Search using BM25
    

    def sparse_search(

        self,

        query,

        top_k=5

    ):

        print("\nPerforming Sparse Search...")

        sparse_results = []

        
        # Tokenize Query
        

        query_tokens = query.lower().split()

       
        # Read Every BM25 Index
        

        bm25_files = sorted(

            self.bm25_directory.glob(

                "*_bm25.pkl"

            )

        )

        for bm25_file in bm25_files:

            
            # Load BM25 Index
           

            with open(

                bm25_file,

                "rb"

            ) as file:

                bm25 = pickle.load(

                    file

                )

            
            # Load Metadata
            

            metadata_file = bm25_file.with_name(

                bm25_file.name.replace(

                    "_bm25.pkl",

                    "_metadata.json"

                )

            )

            with open(

                metadata_file,

                "r",

                encoding="utf-8"

            ) as file:

                metadata = json.load(

                    file

                )

            
            # Calculate BM25 Scores
            

            scores = bm25.get_scores(

                query_tokens

            )

            
            # Get Top Results
            

            top_indices = sorted(

                range(

                    len(scores)

                ),

                key=lambda i: scores[i],

                reverse=True

            )[:top_k]

            for index in top_indices:

                sparse_results.append(

                    {

                        "chunk_id": metadata[index]["chunk_id"],

                        "text": metadata[index]["text"],

                        "video_id": metadata[index]["video_id"],

                        "type": metadata[index]["type"],

                        "timestamp": metadata[index]["timestamp"],

                        "score": float(

                            scores[index]

                        )

                    }

                )

        
        # Sort All Results
        

        sparse_results = sorted(

            sparse_results,

            key=lambda x: x["score"],

            reverse=True

        )

        print(

            f" Retrieved {len(sparse_results[:top_k])} Sparse Results"

        )

        return sparse_results[:top_k]
    
    # Hybrid Retrieval
    

    def hybrid_search(

        self,

        query,

        dense_k=5,

        sparse_k=5

    ):

        
        # Dense Retrieval
        

        dense_results = self.dense_search(

            query,

            top_k=dense_k

        )

        
        # Sparse Retrieval
        

        sparse_results = self.sparse_search(

            query,

            top_k=sparse_k

        )

       
        # Merge Results
        

        hybrid_results = {}

        # Dense Results

        for result in dense_results:

            chunk_id = result["chunk_id"]

            hybrid_results[chunk_id] = result

            hybrid_results[chunk_id]["retrieval"] = "Dense"

        # Sparse Results

        for result in sparse_results:

            chunk_id = result["chunk_id"]

            if chunk_id in hybrid_results:

                hybrid_results[chunk_id]["retrieval"] = "Hybrid"

            else:

                hybrid_results[chunk_id] = result

                hybrid_results[chunk_id]["retrieval"] = "Sparse"

        
        # Convert Dictionary to List
        

        final_results = list(

            hybrid_results.values()

        )

        print(

            f"\nTotal Hybrid Results : {len(final_results)}"

        )

        return final_results
   
    # Process User Query
    

    def process_query(

        self,

        query

    ):

        print("\n========================================")
        print("Hybrid Retrieval Started")
        print("========================================")

        print(f"\nQuery : {query}")

        results = self.hybrid_search(

            query

        )

        print("\n========================================")
        print("Retrieved Chunks")
        print("========================================")

        for index, result in enumerate(

            results,

            start=1

        ):

            print(f"\nRank : {index}")

            print(f"Chunk ID   : {result['chunk_id']}")

            print(f"Video ID   : {result['video_id']}")

            print(f"Type       : {result['type']}")

            print(f"Timestamp  : {result['timestamp']}")

            print(f"Source     : {result['retrieval']}")

            print(f"Score      : {result['score']}")

            print(f"Text       :\n{result['text'][:250]}...")

        return results



# Main


if __name__ == "__main__":

    retriever = HybridRetriever()

    while True:

        print("\n========================================")
        print("Hybrid Text RAG")
        print("========================================")

        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":

            print("\nExiting Hybrid Retriever...")

            break

        retriever.process_query(

            query

        )