import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from loaders import TextFileLoader, DocxFileLoader, XlsxFileLoader, PyMuPDFLoader, ImageFileLoader
from retriever import EnhancedRetriever
from utils import generate_rephrased_questions, create_detailed_context
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Apply the nest_asyncio patch to allow nested event loops in Jupyter
nest_asyncio.apply()

# Load environment variables from a .env file
load_dotenv()

# Get the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model configuration
MODEL = "llama3.1"  # Change this to "gpt-4", "gpt-3.5-turbo", "llama3.1", etc.

# Initialize the model and embeddings based on the chosen MODEL
if MODEL.startswith("gpt"):
    # Use OpenAI's GPT models
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
else:
    # Use Ollama models
    model = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)

# Function to load documents from a directory and its subdirectories
def load_documents(directory):
    all_pages = []
    accepted_documents = 0
    rejected_documents = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.pdf'):
                    loader = PyMuPDFLoader(file_path)
                elif file.lower().endswith('.txt'):
                    loader = TextFileLoader(file_path)
                elif file.lower().endswith('.docx'):
                    loader = DocxFileLoader(file_path)
                elif file.lower().endswith('.xlsx'):
                    loader = XlsxFileLoader(file_path)
                elif file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    loader = ImageFileLoader(file_path)
                else:
                    continue
                pages = loader.load_and_split()
                if pages:
                    all_pages.extend([Document(page_content=page) for page in pages])
                    accepted_documents += 1
                else:
                    print(f"No content extracted from {file_path}.")
                    rejected_documents += 1
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                rejected_documents += 1

    print(f"Accepted documents: {accepted_documents}")
    print(f"Rejected documents: {rejected_documents}")
    return all_pages

# Define a few-shot learning example
few_shot_examples = """
Example 1:
Context: The Singapore Institute of Technology (SIT) is Singapore’s fifth autonomous university and offers applied degree programmes focused on science and technology.
Question: What is the focus of SIT's degree programmes?
Answer: The Singapore Institute of Technology (SIT) focuses on applied degree programmes in science and technology.

Example 2:
Context: Fintech companies in Singapore are subject to regulations set by the Monetary Authority of Singapore (MAS), which oversees the financial sector and ensures its stability.
Question: What authority oversees fintech regulations in Singapore?
Answer: The Monetary Authority of Singapore (MAS) oversees fintech regulations in Singapore.
"""

# Define a prompt template with few-shot examples
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=few_shot_examples + "\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
)

# Custom function to chain the components
async def custom_chain(question):
    # Retrieve context documents dynamically using EnhancedRetriever
    retriever = EnhancedRetriever(vectorstore)
    context_documents = await retriever.get_relevant_documents(question)

    # Combine content of retrieved documents
    context = "\n".join([doc.page_content for doc in context_documents])

    # Create a detailed context with rephrased questions
    detailed_context = create_detailed_context(context, question)

    # Format the prompt with the detailed context
    prompt = prompt_template.format(context=detailed_context, question=question)

    # Ensure the prompt is passed as a list
    response = await asyncio.to_thread(model.generate, [prompt])

    return response



# Asynchronous querying
async def main_chain():
    questions = [
        "Tell me about clusters of SIT",
        # You can add more questions here
    ]

    for question in questions:
        answer = await custom_chain(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")

# Specify the directory containing your documents
directory = r'D:\my_rag_project\Data'  # Replace with your directory path

# Load documents
pages = load_documents(directory)
# Check if any pages were loaded
if not pages:
    print("No documents found. Please check the directory path and file formats.")
else:
    # Define batch size to keep under token limit (adjust based on your use case)
    batch_size = 50

    # Load existing FAISS vector store if available
    index_file = 'faiss_index'

    if os.path.exists(index_file):
        vectorstore = FAISS.load_local(index_file, embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing FAISS vector store.")
    else:
        print("No existing FAISS vector store found, creating a new one.")
        # Create a new vector store from documents
        vectorstore = FAISS.from_documents(pages, embedding=embeddings)
        vectorstore.save_local(index_file)
        print("Created and saved new FAISS vector store.")

    # If you have new documents to add, you can use:
    # vectorstore.add_documents(new_documents)

    # Integrate chat history into the vector store
    chat_history = ["Hi, how are you?", "What is the latest update on AI research?"]  # Example chat history
    chat_documents = [Document(page_content=message) for message in chat_history]
    vectorstore.add_documents(chat_documents)
    print("Integrated chat history into the vector store.")

    # Save the updated vector store
    vectorstore.save_local(index_file)
    print("Updated FAISS vector store saved.")

    # Run the main_chain function
    asyncio.run(main_chain())
