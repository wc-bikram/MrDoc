from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# ✅ Updated imports for Hugging Face
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline  # Recommended version

app = Flask(__name__)
load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load Hugging Face Embeddings
embeddings = download_hugging_face_embeddings()

# Connect to existing Pinecone index
index_name = "medibot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever setup
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Load FLAN-T5 base (lightweight for CPU)
hf_pipeline = pipeline(
    "text2text-generation",                   # Use 'text2text-generation' for FLAN models
    model="google/flan-t5-base",              # Smaller model for fast local usage
    device=-1,                                # CPU only (set to "auto" if GPU available)
    max_new_tokens=256,
    do_sample=True,                           # Enable temperature control
    temperature=0.4
)

# Wrap in LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)
    try:
        response = rag_chain.invoke({"input": msg})
        print("Bot:", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print("Error:", e)
        return "❌ Error processing your request. Try again later."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)