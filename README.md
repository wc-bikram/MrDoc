# 🩺 Medibot – Medical Chatbot using RAG, Hugging Face & Pinecone

Medibot is an AI-powered medical assistant chatbot that can intelligently answer health-related questions by retrieving and synthesizing relevant information from medical documents using **Retrieval-Augmented Generation (RAG)**.

![Medibot UI](assets/medibot_ui_screenshot.png) <!-- Replace with actual image path if hosted -->

---

## 🚀 Features

- 🤖 **RAG-based Chatbot**: Uses LangChain’s Retrieval-Augmented Generation (RAG) to find answers from medical texts.
- 📄 **PDF Knowledge Base**: Upload and embed chunks from medical documents.
- 🧠 **LLM Powered**: Runs on `google/flan-t5-base` using Hugging Face Transformers.
- 📦 **Pinecone Integration**: For fast, scalable vector search.
- 🌐 **Flask UI**: Clean frontend interface for chatting.
- 💡 **Instruction-Tuned Prompts**: Boosts the chatbot’s relevance and quality of answers.

---

## 🧰 Tech Stack

| Tool | Role |
|------|------|
| **Python** | Core language |
| **Flask** | Web app server |
| **LangChain** | RAG pipeline |
| **Hugging Face Transformers** | For FLAN-T5 inference |
| **sentence-transformers** | Text embedding (MiniLM-L6-v2) |
| **Pinecone** | Vector DB for retrieval |
| **HTML/CSS** | Frontend UI |

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/wc-bikram/MrDoc.git
cd medibot

2. Create a virtual environment and install dependencies
 
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

3. Set up your .env file

PINECONE_API_KEY=your-pinecone-api-key

4. Embed and upload PDF data
use the provided notebook or Python script to:

Load medical PDFs

Chunk and embed text

Upsert vectors to your Pinecone index

5. Run the app
bash
Copy
Edit
python app.py 


Then visit: http://localhost:5000

6.  📂 Project Structure
bash
Copy
Edit
medibot/
├── app.py                      # Main Flask app
├── requirements.txt
├── .env                        # API keys
├── templates/
│   └── index.html              # Chat UI
├── static/
│   └── style.css               # UI styling
├── src/
│   ├── helper.py               # PDF loader and embedder
│   └── prompt.py               # System prompt used in the chatbot
└── notebook/
    └── vectorstore_setup.ipynb # Preprocessing and Pinecone upload

7. 📜 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Hugging Face Transformers

LangChain

Pinecone

Mistral & FLAN-T5 Models

Built with ❤️ by Bikram Roy.
