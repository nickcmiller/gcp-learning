import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.callbacks.base import BaseCallbackHandler

openai_api_key = os.getenv('OPENAI_API_KEY')

# Custom callback handler to capture streamed responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.content = []

    def on_new_token(self, token: str):
        self.content.append(token)

    def get_content(self):
        return ''.join(self.content)

def generate_response(input_text:str) -> str:
    handler = StreamHandler()
    llm = OpenAI(
        temperature=0.5, 
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[handler]
    )
    llm(input_text)
    return handler.get_content()

st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    response = generate_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})