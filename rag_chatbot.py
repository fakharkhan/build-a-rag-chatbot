from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
import time

# Chunk the document based on h2 headers.
with open("WonderVector5000.md", "r") as f:
    markdown_document = f.read()

headers_to_split_on = [
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = markdown_splitter.split_text(markdown_document)

# Initialize HuggingFace embeddings
model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Initialize Pinecone and create index
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-getting-started"

# Delete the existing index if it has the wrong dimension
if index_name in pc.list_indexes().names():
    index_info = pc.describe_index(index_name)
    if index_info.dimension != 768:  # Check if dimension is incorrect
        pc.delete_index(index_name)  # Delete the index
        print(f"Deleted existing index '{index_name}' due to dimension mismatch.")

# Create a new index with the correct dimension
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Correct dimension for the model
        metric="cosine",
        spec=spec
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# See that it is empty
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")

# Embed and upsert chunks into Pinecone
namespace = "wondervector5000"

print(f"Upserting {len(md_header_splits)} documents into namespace '{namespace}'...")
docsearch = PineconeVectorStore.from_documents(
    documents=md_header_splits,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

# Wait for upsert to complete
time.sleep(5)

# Verify upsert
index = pc.Index(index_name)
stats = index.describe_index_stats()
print("Index after upsert:")
print(stats)

# Check if namespace exists in stats
if namespace in stats['namespaces']:
    print(f"Namespace '{namespace}' has {stats['namespaces'][namespace]['vector_count']} vectors.")
else:
    print(f"Namespace '{namespace}' not found in index stats. Upsert may have failed.")
print("\n")
time.sleep(2)

# List and query records in the namespace
print("Querying records in namespace...")

# Use query to retrieve records
query_response = index.query(
    vector=[0] * 768,  # Dummy vector of zeros
    namespace=namespace,
    top_k=15,  # Retrieve all vectors in the namespace
    include_values=True,
    include_metadata=True
)

# Print query results
for match in query_response['matches']:
    print("Query Result:")
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")
    print(f"Values: {match['values'][:5]}...")  # Print first 5 values for brevity
    print("\n")