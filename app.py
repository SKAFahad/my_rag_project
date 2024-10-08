from flask import Flask, request, jsonify, render_template_string
import asyncio
from main import custom_chain

app = Flask(__name__)

# HTML template for the web interface
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Model Web App</title>
</head>
<body>
    <h1>Welcome to the RAG Model Web App!</h1>
    <form action="/query" method="post">
        <label for="question">Enter your question:</label><br><br>
        <input type="text" id="question" name="question" required><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if answer %}
    <h2>Answer:</h2>
    <p>{{ answer }}</p>
    <form action="/query" method="post">
        <label for="question">Ask another question:</label><br><br>
        <input type="text" id="question" name="question" required><br><br>
        <input type="submit" value="Submit">
    </form>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/query', methods=['POST'])
async def query():
    question = request.form['question']
    answer = await custom_chain(question)
    return render_template_string(html_template, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template_string
import asyncio
from main import custom_chain

app = Flask(__name__)

# HTML template for the web interface
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Model Web App</title>
</head>
<body>
    <h1>Welcome to the RAG Model Web App!</h1>
    <form action="/query" method="post">
        <label for="question">Enter your question:</label><br><br>
        <input type="text" id="question" name="question" required><br><br>
        <input type="submit" value="Submit">
    </form>
    {% if answer %}
    <h2>Answer:</h2>
    <p>{{ answer }}</p>
    <form action="/query" method="post">
        <label for="question">Ask another question:</label><br><br>
        <input type="text" id="question" name="question" required><br><br>
        <input type="submit" value="Submit">
    </form>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/query', methods=['POST'])
async def query():
    question = request.form['question']
    answer = await custom_chain(question)
    return render_template_string(html_template, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
