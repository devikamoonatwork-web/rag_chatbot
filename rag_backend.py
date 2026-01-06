import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


def get_qa_chain():
    # ----------------------------
    # PDF PATHS
    # ----------------------------
    pdf_paths = [
        "/Users/devika/Desktop/rag_chatbot/data/training_study_material.pdf",
        "/Users/devika/Desktop/rag_chatbot/data/user_definitions_and_access.pdf",
        "/Users/devika/Desktop/rag_chatbot/data/wiscoreia_mentor_buddy.pdf"
    ]

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # ----------------------------
    # LOAD PDFs WITH METADATA
    # ----------------------------
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        pdf_name = os.path.basename(pdf_path)
        for doc in docs:
            doc.metadata["source_pdf"] = pdf_name
        all_docs.extend(docs)

    # ----------------------------
    # SPLIT DOCUMENTS
    # ----------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(all_docs)

    # ----------------------------
    # EMBEDDINGS + VECTOR STORE
    # ----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )

    # ----------------------------
    # LOCAL LLM
    # ----------------------------
    llm = OllamaLLM(model="llama3.2")

    # ----------------------------
    # QUESTION REPHRASER (FOLLOW-UP FIX)
    # ----------------------------
    question_rephrase_template = """
You are a question rephraser.

Rules:
- If the question is already standalone, return it EXACTLY.
- If the question refers to previous context ("that", "it", "this cost"),
  resolve the reference using chat history.
- Preserve numeric intent, entity names, and definitions EXACTLY.
- Do NOT broaden or reinterpret the question.

Chat History:
{chat_history}

Follow-up Question:
{question}

Standalone Question:
"""

    question_prompt = PromptTemplate(
        template=question_rephrase_template,
        input_variables=["chat_history", "question"]
    )

    question_generator = LLMChain(
        llm=llm,
        prompt=question_prompt
    )

    # ----------------------------
    # MAIN QA PROMPT (CORE FIX)
    # ----------------------------
    qa_template = """
You are a highly precise document-grounded assistant.

You answer questions using ONLY the provided PDF context.
You may use chat history ONLY to understand references.

=====================
HOW TO USE CHAT HISTORY
=====================
- Use chat history only to resolve references (e.g., "that cost").
- Do NOT repeat or summarize past answers.
- Answer ONLY the current question.

=====================
CRITICAL ACCURACY RULES
=====================
1. NEVER invent information.
2. NEVER infer relationships unless explicitly stated in the same sentence.
3. NEVER reformat or normalize numeric values.

=====================
NUMERIC VALUE RULE (MANDATORY)
=====================
If the answer includes numbers (prices, ranges, dates, quantities):
- Copy EXACT text from the context
- Preserve symbols, commas, spacing, and dashes
- Example:
  "$6,500 – $8,500" ✅
  "$6500 to $8500" ❌

If exact numeric text is not found, say:
"I don't have enough information about that in the PDF documents."

=====================
DEFINITION RULE
=====================
If asked "What is X":
- Find the definition exactly as written
- Quote the FULL definition
- Do NOT paraphrase or shorten

=====================
SEARCH FAILURE RULE
=====================
Before saying you lack information:
- Re-check all retrieved chunks
- Look in headings, bullet lists, tables
- Only then respond with lack of information

=====================
CONTEXT
=====================
{context}

=====================
CHAT HISTORY (REFERENCE ONLY)
=====================
{chat_history}

=====================
CURRENT QUESTION
=====================
{question}

=====================
FINAL ANSWER
=====================
Answer ONLY the question above.
"""

    qa_prompt = PromptTemplate(
        template=qa_template,
        input_variables=["context", "chat_history", "question"]
    )

    from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain_classic.chains.llm import LLMChain as ClassicLLMChain

    doc_chain = ClassicLLMChain(llm=llm, prompt=qa_prompt)

    combine_docs_chain = StuffDocumentsChain(
        llm_chain=doc_chain,
        document_variable_name="context"
    )

    qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=combine_docs_chain,
        return_source_documents=True,
        verbose=False
    )

    return qa_chain


# ----------------------------
# INITIALIZE GLOBAL CHAIN
# ----------------------------
chain = None
try:
    print("🚀 Initializing Local RAG Chatbot...")
    chain = get_qa_chain()
    print("✅ RAG Chatbot Ready")
except Exception as e:
    print(f"❌ Initialization error: {e}")


# ----------------------------
# ASK QUESTION FUNCTION
# ----------------------------
def ask_question(query, chat_history=None):
    if chain is None:
        return "The AI system failed to start."

    query_clean = query.strip()
    query_lower = query_clean.lower()

    # ----------------------------
    # SMART GREETINGS
    # ----------------------------
    greetings = ["hello", "hi", "hey", "how are you", "what's up", "greetings"]
    if any(g in query_lower for g in greetings):
        return (
            "Hi there! 😊 I'm here to help you get accurate answers directly "
            "from your documents. What would you like to know?"
        )

    # ----------------------------
    # USER DISSATISFACTION HANDLING
    # ----------------------------
    unsatisfied_markers = [
        "wrong",
        "incorrect",
        "not helpful",
        "you already said",
        "not accurate",
        "that's not what i asked"
    ]

    if any(m in query_lower for m in unsatisfied_markers):
        return (
            "You're right to point that out — I’m sorry about that. 🙏 "
            "Let me carefully re-check the documents and answer again."
        )

    # ----------------------------
    # FORMAT CHAT HISTORY
    # ----------------------------
    formatted_history = []
    if chat_history:
        i = 0
        while i < len(chat_history):
            msg = chat_history[i]
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    ai_msg = ""
                    if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "assistant":
                        ai_msg = chat_history[i + 1].get("content", "")
                        i += 2
                    else:
                        i += 1
                    formatted_history.append((user_msg, ai_msg))
                else:
                    i += 1
            elif isinstance(msg, tuple) and len(msg) == 2:
                formatted_history.append(msg)
                i += 1
            else:
                i += 1

    # Keep last 6 exchanges for better memory
    formatted_history = formatted_history[-6:]

    # ----------------------------
    # INVOKE CHAIN
    # ----------------------------
    response = chain.invoke({
        "question": query_clean,
        "chat_history": formatted_history
    })

    if isinstance(response, dict):
        answer = response.get("answer", "").strip()
        if not answer:
            return "I couldn't find relevant information in the PDF documents."
        return answer

    return str(response)

