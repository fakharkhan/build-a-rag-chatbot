from flask import Flask, render_template, request, jsonify
from Chatbot import Chatbot, setup_pinecone
from langchain_openai import ChatOpenAI
import os

app = Flask(__name__)

# Initialize components
docsearch = setup_pinecone()
bot = Chatbot(docsearch)
llm = ChatOpenAI(
    openai_api_key=os.environ.get('OPENAI_API_KEY'),
    model_name='gpt-4',
    temperature=0.0
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    use_knowledge = data.get('use_knowledge', False)
    
    try:
        if use_knowledge:
            response = bot.chat(query)
        else:
            response = llm.invoke(query).content
            
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 