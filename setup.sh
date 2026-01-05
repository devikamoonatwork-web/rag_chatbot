import re
import os
import json
from pathlib import Path

# Updated imports for LangChain 0.2+ / 0.3 compatibility
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Please run: pip install langchain langchain-community langchain-ollama faiss-cpu pypdf")
    raise

# =============================
# Configuration
# =============================

VECTOR_DIR = "vectorstore"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# =============================
# Initialize Components
# =============================

print("🔮 Initializing Ollama embeddings and LLM...")

# Note: Ensure Ollama is running locally with 'llama3.2' pulled
embeddings = OllamaEmbeddings(model="llama3.2")
llm = ChatOllama(model="llama3.2", temperature=0.1)

# =============================
# Load or Create Vectorstore
# =============================

def load_or_create_vectorstore():
    """Load existing vectorstore or create new one from PDFs."""
    index_path = Path(VECTOR_DIR) / "index.faiss"
    
    if os.path.exists(VECTOR_DIR) and index_path.exists():
        print(f"📂 Loading existing vectorstore from {VECTOR_DIR}...")
        # allow_dangerous_deserialization is required for loading local FAISS files
        vectorstore = FAISS.load_local(
            VECTOR_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✅ Vectorstore loaded successfully")
        return vectorstore
    else:
        print(f"📄 Creating new vectorstore from PDFs in {DATA_DIR}...")
        
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
        if not pdf_files:
            # Create a dummy file or warning instead of crashing immediately
            print(f"⚠️ Warning: No PDF files found in {DATA_DIR}/")
            return None
        
        all_documents = []
        document_titles = {}
        
        for pdf_path in pdf_files:
            print(f"  Loading: {pdf_path.name}")
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["source"] = pdf_path.name
                    doc.metadata["title"] = pdf_path.stem
                
                document_titles[pdf_path.name] = pdf_path.stem
                all_documents.extend(docs)
            except Exception as e:
                print(f"  ❌ Error loading {pdf_path.name}: {e}")

        if not all_documents:
            return None

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(all_documents)
        
        # Create and Save
        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        vectorstore.save_local(VECTOR_DIR)
        
        # Save titles mapping
        with open(Path(VECTOR_DIR) / "document_titles.json", 'w') as f:
            json.dump(document_titles, f, indent=2)
            
        return vectorstore

# Global Vectorstore Instance
vectorstore = load_or_create_vectorstore()

# =============================
# Utility Functions
# =============================

def get_available_documents():
    titles_file = Path(VECTOR_DIR) / "document_titles.json"
    if titles_file.exists():
        with open(titles_file, 'r') as f:
            return sorted(list(json.load(f).values()))
    return []

# =============================
# Prompt Template
# =============================

PROMPT = ChatPromptTemplate.from_template("""
You are an assistant. Answer based ONLY on the context and history.
Available docs: {available_documents}
History: {chat_history}
Context: {context}
Question: {question}
Answer:""")

# =============================
# Main RAG Function
# =============================

def ask_question(question: str, chat_history=None, debug=False) -> str:
    if vectorstore is None:
        return "The knowledge base is empty. Please add PDFs to the 'data' folder."

    question_clean = question.strip()
    
    # Simple Greeting Check
    if re.match(r"^(hi|hello|hey|hii)$", question_clean.lower()):
        docs = get_available_documents()
        return f"Hello! I can help with: {', '.join(docs)}" if docs else "Hello! Add some PDFs so I can help you."

    try:
        # Retrieval
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question_clean)
        
        context = "\n\n".join([f"[{d.metadata.get('title')}] {d.page_content}" for d in docs])
        
        # History formatting
        history_text = ""
        if chat_history:
            for msg in chat_history[-5:]:
                history_text += f"{msg['role']}: {msg['content']}\n"

        # Logic
        chain = PROMPT | llm
        response = chain.invoke({
            "available_documents": ", ".join(get_available_documents()),
            "chat_history": history_text or "None",
            "context": context,
            "question": question_clean
        })
        
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"

print("--- ✅ RAG Backend initialized successfully! ---")