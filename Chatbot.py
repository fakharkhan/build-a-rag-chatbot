from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langsmith import Client
import os
import time
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize LangSmith client
client = Client(api_key=os.environ.get("LANGSMITH_API_KEY"))

def setup_pinecone():
    # Initialize Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Initialize embeddings
    model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Set up index name and namespace
    index_name = "rag-getting-started"
    namespace = "wondervector5000"
    
    # Get the Pinecone index
    index = pc.Index(index_name)
    
    # Create the PineconeVectorStore
    docsearch = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
    return docsearch

class Chatbot:
    def __init__(self, docsearch):
        # Initialize the LLM with OpenAI
        self.llm = ChatOpenAI(
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            model_name='gpt-4o-mini',
            temperature=0.0
        )
        
        # Set up the retrieval chain
        self.retriever = docsearch.as_retriever()
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        
        # Create document chain and retrieval chain
        self.combine_docs_chain = create_stuff_documents_chain(
            self.llm, self.retrieval_qa_chat_prompt
        )
        self.retrieval_chain = create_retrieval_chain(
            self.retriever, self.combine_docs_chain
        )

    def chat(self, query):
        """Send a query to the chatbot and get a response"""
        response = self.retrieval_chain.invoke({"input": query})
        return response['answer']

# Initialize the LLM separately for comparison
llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

# Define the query
query1 = "What are the first 3 steps for getting started with the WonderVector5000?"

# Get answer without Pinecone knowledge
answer1_without_knowledge = llm.invoke(query1)

# Print results
print("Query 1:", query1)
print("\nAnswer without knowledge:\n\n", answer1_without_knowledge.content)
print("\n")
time.sleep(2)

# Initialize the Chatbot with Pinecone context
docsearch = setup_pinecone()
bot = Chatbot(docsearch)

# Get answer with Pinecone knowledge
answer1_with_knowledge = bot.chat(query1)

# Print results
print("Answer with knowledge:\n\n", answer1_with_knowledge)
print("\n")
time.sleep(2)
