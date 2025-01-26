from pinecone import Pinecone
import os
import logging

# Constants
INDEX_NAME = "rag-getting-started"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def delete_pinecone_index():
    """Delete the Pinecone index used in the application"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        # Check if index exists
        if INDEX_NAME in pc.list_indexes().names():
            # Delete the index
            pc.delete_index(INDEX_NAME)
            logger.info(f"Successfully deleted index '{INDEX_NAME}'")
        else:
            logger.info(f"Index '{INDEX_NAME}' does not exist")
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}")
        raise

if __name__ == "__main__":
    delete_pinecone_index() 