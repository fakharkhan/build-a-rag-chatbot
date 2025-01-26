from pinecone import Pinecone
import os

def delete_pinecone_index():
    """Delete the Pinecone index used in the application"""
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Index name to delete
    index_name = "rag-getting-started"
    
    # Check if index exists
    if index_name in pc.list_indexes().names():
        # Delete the index
        pc.delete_index(index_name)
        print(f"Successfully deleted index '{index_name}'")
    else:
        print(f"Index '{index_name}' does not exist")

if __name__ == "__main__":
    delete_pinecone_index() 