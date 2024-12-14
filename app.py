
import faiss
import pickle
import openai
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS 
from config import OPENAI_API_KEY

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

# Load the embedding model and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.index")
with open("data_chunks.pkl", "rb") as f:
    data_chunks = pickle.load(f)

# Define function to query OpenAI
def query_openai(prompt, context):
    combined_prompt = f"Use the following context to answer the question:\n\nContext: {context}\n\nQuestion: {prompt}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
You are EH-GPT, an AI assistant designed to help students and teachers understand and learn from the Sindh Board Textbook. 
Instructions:
1. Always provide clear answers based on the provided textbook content.
2. Reference the relevant page number when applicable.
3. Use simple, student-friendly language.
4. If a concept needs further explanation, break it down into steps or examples.
5. When the user asks for definitions or explanations, include real-world examples when possible.

Behavior:
- If the question is unclear, politely ask for clarification.
- If a user asks general questions (e.g., greetings, well-being, or casual chats), respond politely and engagingly.
You are EH-GPT, an AI assistant designed to assist with educational queries and general day-to-day conversations. 
When a user asks general questions (e.g., greetings, well-being, or casual chats), respond politely and engagingly. Always maintain a friendly and helpful tone.
"""
                  },
                  {
                      "role": "user", 
                      "content": combined_prompt
                  }
            ]
    )
    return response['choices'][0]['message']['content']

MAX_CHUNK_LENGTH = 500

# Define function to search and answer
def search_and_answer(query):
    query_embedding = embedding_model.encode([query])
    
    # Get the total number of items in the index
    total_items = index.ntotal
    
    # Limit to 5 closest items
    k = min(total_items, 5)
    
    # Retrieve the nearest neighbors (k closest items)
    D, I = index.search(query_embedding, k=k)
    
    # Combine the context from the k nearest neighbors, truncated to MAX_CHUNK_LENGTH
    context = "\n".join([data_chunks[i][:MAX_CHUNK_LENGTH] for i in I[0]])
    
    return query_openai(query, context)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Define the POST route for querying the bot
@app.route('/ask', methods=['POST'])
def ask_bot():
    try:
        # Get the question from the request
        question = request.json.get('question')
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Call the search_and_answer function
        answer = search_and_answer(question)
        
        # Return the response as JSON
        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host= "0.0.0.0", port = 5000)
