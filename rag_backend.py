import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# UPDATED: We now import ConversationalRetrievalChain for conversation memory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

def get_qa_chain():
    # List of all PDF paths to load
    pdf_paths = [
        "/Users/devika/Desktop/rag_chatbot/data/training_study_material.pdf",
        "/Users/devika/Desktop/rag_chatbot/data/user_definitions_and_access.pdf",
        "/Users/devika/Desktop/rag_chatbot/data/wiscoreia_mentor_buddy.pdf"
    ]
    
    # Verify all PDFs exist
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # 1. Load all PDFs and combine documents with metadata
    all_docs = []
    for pdf_path in pdf_paths:
        print(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # Add source metadata to each document
        pdf_name = os.path.basename(pdf_path)
        for doc in docs:
            doc.metadata['source_pdf'] = pdf_name
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from {pdf_name}")
    
    print(f"Total pages loaded: {len(all_docs)}")
    
    # 2. Split Text - Increased chunk size to preserve more context
    # Larger chunks help preserve complete sections like "Phase 3: Go-to-Market Layer"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased from 600 to preserve more context
        chunk_overlap=200  # Increased overlap to ensure continuity
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"Split into {len(splits)} chunks from all PDFs")
    
    # 3. Local Embedding Model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. Create Local Vector Store
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("Vector store created successfully")
    
    # 5. Local LLM (Ollama llama3.2)
    print("Initializing LLM...")
    llm = OllamaLLM(model="llama3.2")
    
    # 6. Create Chain (using the classic RetrievalQA)
    # Configure retriever to return more relevant chunks from all PDFs
    # Using MMR (Maximal Marginal Relevance) for better diversity across PDFs
    # This helps ensure we get results from different sections/documents
    # Use similarity search with more chunks to ensure we find definitions
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Similarity search for more accurate results
        search_kwargs={
            "k": 12  # Retrieve top 12 chunks to ensure we find definitions and relevant content
        }
    )
    
    print("Creating Conversational QA chain with memory...")
    
    # Template for generating standalone question from chat history
    question_generator_template = """You are a question rephraser. Your job is to convert follow-up questions into standalone questions.

IMPORTANT RULES:
1. If the follow-up question is already clear and standalone, return it EXACTLY as is
2. Only rephrase if the question references something from chat history (like "it", "that", "the above", "them", "this")
3. When rephrasing, use the ACTUAL topic from chat history, not generic terms
4. DO NOT add connections or relationships that aren't in the original question
5. If the question asks about multiple things (e.g., "what is X and what is Y"), keep both parts separate
6. For "what is X" questions, keep them as "what is X" - do not change the format

Chat History:
{chat_history}

Follow Up Question: {question}

Your task: If the question needs context from chat history, rephrase it to be standalone. Otherwise, return it EXACTLY as is.

Standalone Question:"""
    
    question_prompt = PromptTemplate(
        template=question_generator_template,
        input_variables=["chat_history", "question"]
    )
    question_generator = LLMChain(llm=llm, prompt=question_prompt)
    
    # Custom prompt template for answering with context and chat history
    qa_template = """You are an assistant that answers questions based ONLY on the context provided from multiple PDF documents. 
The context below contains information from three PDF documents:
1. training_study_material.pdf
2. user_definitions_and_access.pdf  
3. wiscoreia_mentor_buddy.pdf

CRITICAL RULES - FOLLOW THESE EXACTLY:
1. Answer ONLY the CURRENT question asked. Do NOT repeat answers from previous questions.
2. DO NOT make ANY connections between topics unless the context EXPLICITLY states a direct connection in the same sentence or paragraph.
3. DO NOT say "used by", "for", "related to" unless the context EXPLICITLY states that relationship.
4. If asked "what is X", find the definition or description of X in the context and quote it directly. If X is in a list, describe it as it appears in that list.
5. For DEFINITIONS and DESCRIPTIONS, quote them EXACTLY as they appear in the context. Do not rephrase or add connections.
6. For NUMERIC VALUES (costs, prices, estimates, dollar amounts, percentages, dates, quantities):
   - Quote them EXACTLY as they appear in the context
   - Use the exact format: e.g., "$6,500 – $8,500" not "$6500 to $8500"
   - Preserve all formatting, commas, dashes, and symbols
7. If the context doesn't contain clear information about the question, say "I don't have enough information about that in the PDF documents."
8. DO NOT infer relationships. If Twilio appears in a list and Flipper appears elsewhere, DO NOT say "Twilio is used by Flippers" unless the context explicitly states that.

Context from PDF documents:
{context}

Chat History (for reference only - answer the CURRENT question):
{chat_history}

CURRENT Question to Answer: {question}

Answer Instructions - READ CAREFULLY:
- Answer ONLY the question: {question}
- SEARCH THOROUGHLY through ALL the context provided above to find the answer
- For "what is X" questions, look for:
  1. Definition sections (e.g., "X: The...", "X is...", "X - definition")
  2. Descriptions of X
  3. Lists containing X
  4. Any mention of X with explanation
- Find the EXACT text in the context that answers the question
- Quote definitions and descriptions EXACTLY as they appear in the context
- DO NOT add words like "used by", "for", "related to", "by [group]" unless the context EXPLICITLY uses those exact words
- If the question asks "what is X" and X appears in a list (e.g., "Twilio Voice & Messaging" in a list of tools), describe it as it appears in that list. Do NOT say "used by Y" unless the list explicitly says that.
- If the question asks "what is X" and there's a definition section for X (like "X: The..."), quote that definition exactly, including the full description
- DO NOT say "X is not referred to" or "X is not mentioned" - instead, search more carefully. If you truly cannot find X, say "I don't have enough information about that in the PDF documents."
- DO NOT connect topics that appear in different parts of the context unless they are explicitly connected in the same sentence/paragraph
- DO NOT make assumptions about relationships (e.g., if "Twilio" is in a tools list and "Flipper" is in a definitions section, DO NOT say "Twilio is used by Flippers" unless the context explicitly states that)
- If asked about two things separately (e.g., "what is X and what is Y"), answer each separately based on what the context says about each, without connecting them
- Be direct, factual, and quote the context accurately
- If information is truly not in the context after thorough search, say "I don't have enough information about that in the PDF documents."

EXAMPLES:
- If context says "Twilio Voice & Messaging" in a list, answer: "Twilio Voice & Messaging is listed as a communication tool/platform."
- If context has "Flipper: The Renovation Expert & Capital Multiplier. A flipper buys...", quote the ENTIRE definition: "Flipper: The Renovation Expert & Capital Multiplier. A flipper buys distressed or undervalued properties, renovates them, and then sells them for a profit—often within 3 to 6 months."
- DO NOT say "Twilio is used by Flippers" unless the context explicitly states that relationship
- DO NOT say "X is not referred to in section Y" - instead search all sections

Answer:"""
    
    qa_prompt = PromptTemplate(
        template=qa_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create the conversational retrieval chain
    from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain_classic.chains.llm import LLMChain as ClassicLLMChain
    
    # Create the document chain that combines retrieved docs
    doc_chain = ClassicLLMChain(llm=llm, prompt=qa_prompt)
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=doc_chain,
        document_variable_name="context"
    )
    
    # Create conversational retrieval chain
    qa = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=combine_docs_chain,
        return_source_documents=True,
        verbose=False
    )
    
    print("Conversational QA chain created successfully")
    return qa

# Global chain variable
chain = None

try:
    print("--- 🚀 Initializing Local RAG with Conversation Memory (Ollama + HuggingFace) ---")
    chain = get_qa_chain()
    print("--- ✅ SUCCESS: Your local AI with conversation memory is ready! ---")
except Exception as e:
    print(f"--- ❌ INITIALIZATION ERROR: {e} ---")
    chain = None

def ask_question(query, chat_history=None):
    """
    Ask a question with conversation memory.
    
    Args:
        query: The user's question
        chat_history: List of tuples (human_message, ai_message) or list of messages
                     If None, starts a new conversation
    """
    if chain is None:
        return "The local AI failed to start. Check your terminal logs for import errors."
    try:
        # Process query
        processed_query = query.strip().lower()
        
        # Handle greetings and casual conversation
        greetings = ["hello", "hi", "hey", "how are you", "how's it going", "what's up", "greetings"]
        if any(greeting in processed_query for greeting in greetings):
            return "Hello! 👋 I'm doing great, thank you for asking! I'm here to help you find information from your PDF documents. What would you like to know?"
        
        # Reset to original case for processing
        processed_query = query.strip()
        print(f"Processing CURRENT query: {processed_query}")
        
        # Format chat history for the chain
        # Limit chat history to last 4 exchanges to prevent confusion
        # ConversationalRetrievalChain expects chat_history as a list of tuples: [("Human", "question"), ("AI", "answer")]
        if chat_history is None or len(chat_history) == 0:
            formatted_history = []
        else:
            # Convert Streamlit messages format to LangChain format
            formatted_history = []
            i = 0
            while i < len(chat_history):
                msg = chat_history[i]
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        # Look for the next assistant message to pair with
                        if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "assistant":
                            formatted_history.append((content, chat_history[i + 1].get("content", "")))
                            i += 2
                        else:
                            # Unpaired user message, add with empty AI response
                            formatted_history.append((content, ""))
                            i += 1
                    elif role == "assistant":
                        # Skip unpaired assistant messages (shouldn't happen in normal flow)
                        i += 1
                elif isinstance(msg, tuple) and len(msg) == 2:
                    formatted_history.append(msg)
                    i += 1
                else:
                    i += 1
            
            # Limit to last 4 exchanges (8 messages: 4 user + 4 assistant) to prevent confusion
            formatted_history = formatted_history[-4:] if len(formatted_history) > 4 else formatted_history
        
        print(f"Formatted chat history (last {len(formatted_history)} exchanges): {formatted_history[:1] if formatted_history else 'None'}...")  # Debug
        
        # Invoke the chain with chat history (use formatted_history, not original chat_history)
        response = chain.invoke({
            "question": processed_query,
            "chat_history": formatted_history
        })
        
        # Handle different response formats
        if isinstance(response, dict):
            result = response.get("answer", response.get("result", str(response)))
            # Log source documents for debugging
            if "source_documents" in response:
                sources = response["source_documents"]
                print(f"Retrieved {len(sources)} document chunks")
                # Log which PDFs were referenced and show sample content
                pdfs_checked = set()
                for i, doc in enumerate(sources[:3]):  # Show first 3 chunks
                    if hasattr(doc, 'metadata') and 'source_pdf' in doc.metadata:
                        pdfs_checked.add(doc.metadata['source_pdf'])
                    # Print a preview of retrieved content for debugging
                    content_preview = doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
                    print(f"  Chunk {i+1} preview: {content_preview}...")
                if pdfs_checked:
                    print(f"Searched in PDFs: {', '.join(pdfs_checked)}")
        else:
            result = str(response)
        
        # Check if result is empty or indicates no context
        if not result or len(result.strip()) == 0:
            return "I couldn't find relevant information in the PDF documents to answer your question. Please try rephrasing your question or asking about a different topic."
        
        # Post-process to improve answer quality
        result_lower = result.lower()
        
        # Check for confusing "not found" patterns and improve them
        confusing_patterns = [
            "is not referred to",
            "is not mentioned in",
            "no, [topic] is not",
            "is not in the section"
        ]
        
        # If the answer has a confusing "not found" pattern, it might mean retrieval failed
        # But we'll let the prompt handle this - the prompt now says to search thoroughly
        
        print(f"Query processed successfully")
        return result
    except Exception as e:
        error_msg = f"Error during query: {e}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg