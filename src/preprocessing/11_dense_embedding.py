# =============================================================================
# Import Required Libraries
# =============================================================================

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# =============================================================================
# Dense Embedding Generator
# =============================================================================

class DenseEmbeddingGenerator:

    def __init__(self):

        # ---------------------------------------------------------
        # Chunk Folder
        # ---------------------------------------------------------

        self.chunk_directory = Path(

            "data/chunks"

        )

        # ---------------------------------------------------------
        # Output Folder
        # ---------------------------------------------------------

        self.embedding_directory = Path(

            "data/dense_embeddings"

        )

        self.embedding_directory.mkdir(

            parents=True,

            exist_ok=True

        )

        print("Loading Sentence Transformer Model...")

        # ---------------------------------------------------------
        # Embedding Model
        # ---------------------------------------------------------

        self.model = SentenceTransformer(

            "all-MiniLM-L6-v2"

        )

        print("✓ Model Loaded : all-MiniLM-L6-v2")


    # =========================================================================
    # Get All Chunk Files
    # =========================================================================

    def get_chunk_files(self):

        return sorted(

            self.chunk_directory.glob(

                "*_chunks.json"

            )

        )
        # =========================================================================
    # Generate Dense Embedding
    # =========================================================================

    def generate_embedding(

        self,

        text

    ):

        embedding = self.model.encode(

            text,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        return embedding.tolist()
    
        # =========================================================================
    # Process One Chunk File
    # =========================================================================

    def process_chunk_file(

        self,

        chunk_file

    ):

        print(f"\nProcessing {chunk_file.name}")

        # ---------------------------------------------------------
        # Load Chunk File
        # ---------------------------------------------------------

        with open(

            chunk_file,

            "r",

            encoding="utf-8"

        ) as file:

            chunks = json.load(

                file

            )

        embedded_chunks = []

        # ---------------------------------------------------------
        # Generate Embeddings
        # ---------------------------------------------------------

        for index, chunk in enumerate(

            chunks,

            start=1

        ):

            print(

                f"Generating Embedding {index}/{len(chunks)}"

            )

            embedding = self.generate_embedding(

                chunk["text"]

            )

            embedded_chunk = {

                "chunk_id": chunk["chunk_id"],

                "video_id": chunk["video_id"],

                "type": chunk["type"],

                "timestamp": chunk["timestamp"],

                "text": chunk["text"],

                "embedding": embedding

            }

            embedded_chunks.append(

                embedded_chunk

            )
            # ---------------------------------------------------------
    # Save Dense Embeddings
    # ---------------------------------------------------------

        output_file = self.embedding_directory / (

            chunk_file.name.replace(

                "_chunks.json",

                "_dense.json"

            )

        )

        with open(

            output_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                embedded_chunks,

                file,

                indent=4,

                ensure_ascii=False

            )

        print(

            f"✓ Saved {len(embedded_chunks)} Embeddings"

        )

        print(

            f"Output File : {output_file.name}"

        )
        # =========================================================================
    # Process All Chunk Files
    # =========================================================================

    def process_all_chunk_files(

        self

    ):

        chunk_files = self.get_chunk_files()

        print("\n========================================")
        print("Dense Embedding Generation Started")
        print("========================================")

        print(

            f"\nFound {len(chunk_files)} Chunk Files.\n"

        )

        for index, chunk_file in enumerate(

            chunk_files,

            start=1

        ):

            print(

                f"\n[{index}/{len(chunk_files)}] "

                f"Processing {chunk_file.name}"

            )

            self.process_chunk_file(

                chunk_file

            )

        print("\n========================================")
        print("✓ Dense Embedding Generation Completed")
        print("========================================")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    generator = DenseEmbeddingGenerator()

    generator.process_all_chunk_files()
