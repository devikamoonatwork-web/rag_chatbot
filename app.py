import streamlit as st
from rag_backend import ask_question

# Page styling
st.set_page_config(page_title="My PDF Assistant", page_icon="📚")
st.title("📚 Chat with your PDF Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from backend
    with st.chat_message("assistant"):
        with st.spinner("Searching your PDFs..."):
            try:
                answer = ask_question(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")