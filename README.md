# My RAG Project

This project implements a Retrieval-Augmented Generation (RAG) model using various document loaders, a custom retriever, and language models from OpenAI and Ollama. The project includes a web application built with Flask to interact with the RAG model.

## Table of Contents

- Installation
- Usage
- Project Structure
- Environment Variables
- Contributing
- License

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Steps

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your_username/my_rag_project.git
    cd my_rag_project
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    source venv/bin/activate  # On macOS/Linux
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Install FAISS**:
    ```sh
    pip install faiss-cpu
    ```

## Usage

1. **Set up environment variables**:
    - Create a `.env` file in the project root directory and add your OpenAI API key:
      ```env
      OPENAI_API_KEY=your_openai_api_key
      ```

2. **Run the web application**:
    ```sh
    python app.py
    ```

3. **Interact with the web application**:
    - Use tools like Postman or curl to send POST requests to the `/query` endpoint with a JSON payload containing the question.
    - Example:
      ```sh
      curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"question": "Tell me about clusters of SIT"}'
      ```

## Project Structure

├── loaders.py # Custom file loaders for different document types 
├── retriever.py # Enhanced retriever class using BM25 
├── utils.py # Utility functions for generating rephrased questions and detailed context 
├── main.py # Main logic for loading documents, creating vector store, and querying 
├── app.py # Flask web application ├── requirements.txt # List of required packages 
├── .env # Environment variables (not included in the repository) 
├── .gitignore # Git ignore file └── README.md # Project documentation


## Environment Variables

The project requires the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
