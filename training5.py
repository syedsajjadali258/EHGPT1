import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import numpy as np

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_jsonl(jsonl_path):
    """
    Load and process all key-value pairs from a JSONL file.
    """
    data_chunks = []
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)  # Parse JSON object
                # Combine all key-value pairs into a single string
                combined_entry = " ".join([f"{key}: {value}" for key, value in entry.items()])
                data_chunks.append(combined_entry)  # Append the combined string
    except Exception as e:
        print(f"Error reading JSONL: {e}")
    
    return data_chunks

def embed_and_index_text(data_chunks, faiss_index_path="faiss_index.index", data_chunks_path="data_chunks.pkl"):
    """
    Embed all key-value pairs and index them using FAISS.
    Appends new data to the existing index and data chunks.
    """
    # Load existing data chunks and FAISS index if they exist
    if os.path.exists(faiss_index_path) and os.path.exists(data_chunks_path):
        # Load existing data chunks
        with open(data_chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)
        
        # Load existing FAISS index
        index = faiss.read_index(faiss_index_path)
    else:
        # Initialize new index and chunks if they don't exist
        existing_chunks = []
        index = None

    # Combine new data with existing data
    combined_chunks = existing_chunks + data_chunks

    # Generate embeddings for new data
    embeddings = embedding_model.encode(data_chunks)

    # Initialize FAISS index if it doesn't exist
    if index is None:
        dimension = embeddings.shape[1]  # Dimension of the embedding model
        index = faiss.IndexFlatL2(dimension)

    # Add new embeddings to the FAISS index
    index.add(embeddings)

    # Save updated FAISS index and data chunks
    faiss.write_index(index, faiss_index_path)
    with open(data_chunks_path, "wb") as f:
        pickle.dump(combined_chunks, f)

    print(f"Training complete! {len(data_chunks)} new entries indexed successfully.")
    print(f"Total entries indexed: {len(combined_chunks)}")

def train_on_jsonl(jsonl_path):
    """
    Train the model using the JSONL by embedding and indexing its text.
    """
    # Load data from the JSONL file
    data_chunks = load_jsonl(jsonl_path)
    
    # Embed and index the text
    embed_and_index_text(data_chunks)

def remove_entries(entry_indices, faiss_index_path="faiss_index.index", data_chunks_path="data_chunks.pkl"):
    """
    Remove specific entries from the FAISS index and data chunks.
    """
    # Load existing data chunks and FAISS index
    if not (os.path.exists(faiss_index_path) and os.path.exists(data_chunks_path)):
        print("Error: FAISS index or data chunks file not found.")
        return

    with open(data_chunks_path, "rb") as f:
        data_chunks = pickle.load(f)

    index = faiss.read_index(faiss_index_path)

    # Ensure entry_indices are sorted in descending order to avoid index shifting issues
    entry_indices = sorted(entry_indices, reverse=True)

    # Remove the specified entries from data chunks
    for idx in entry_indices:
        if 0 <= idx < len(data_chunks):
            data_chunks.pop(idx)
        else:
            print(f"Index {idx} out of bounds.")

    # Re-embed the remaining data chunks
    embeddings = embedding_model.encode(data_chunks)

    # Create a new FAISS index
    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatL2(dimension)
    new_index.add(embeddings)

    # Save the updated FAISS index and data chunks
    faiss.write_index(new_index, faiss_index_path)
    with open(data_chunks_path, "wb") as f:
        pickle.dump(data_chunks, f)

    print(f"Removed {len(entry_indices)} entries. Total entries indexed: {len(data_chunks)}")

if __name__ == "__main__":
    action = input("Enter 'train' for training OR 'remove' for removing specific entries: ").strip().lower()

    if action == "train":
        jsonl_path = input("Enter the path to your JSONL file for training: ").strip()
        if not os.path.exists(jsonl_path):
            print(f"Error: The file '{jsonl_path}' does not exist.")
        else:
            print("Starting training...")
            train_on_jsonl(jsonl_path)
    elif action == "remove":
        entry_indices = input("Enter the indices of the entries to remove (comma-separated): ").strip()
        entry_indices = list(map(int, entry_indices.split(',')))
        print(f"Removing entries at indices: {entry_indices}")
        remove_entries(entry_indices)
    else:
        print("Invalid action.")
