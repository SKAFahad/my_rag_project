from flask import Flask, request, jsonify
import asyncio
from main import custom_chain

app = Flask(__name__)

@app.route('/query', methods=['POST'])
async def query():
    data = request.json
    question = data.get('question')
    answer = await custom_chain(question)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(debug=True)
