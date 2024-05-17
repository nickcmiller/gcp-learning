import os
import logging
import streamlit as st
from langchain_openai import OpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Retrieve and log the API key (partially) for debugging
openai_api_key = os.getenv('OPENAI_API_KEY')
logger.debug(f"OpenAI API Key: {openai_api_key[:5]}...")

# Custom callback handler to capture streamed responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.content = []

    def on_new_token(self, token: str):
        self.content.append(token)
        logger.debug(f"Received token: {token}")  # Debugging: log each token received

    def get_content(self):
        return ''.join(self.content)

def generate_response(input_text: str) -> str:
    logger.debug(f"Generating response for input: {input_text}")
    handler = StreamHandler()
    llm = OpenAI(
        temperature=0.5, 
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[handler]
    )
    # Make sure the invoke method is used
    try:
        logger.debug("Invoking LLM...")
        llm.invoke(input_text)
        logger.debug("Invocation complete.")
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        return "Error generating response."

    response_content = handler.get_content()
    logger.debug(f"Generated response: {response_content}")  # Debugging: log the final response
    return response_content

# Fallback non-streaming function
def generate_response_non_streaming(input_text: str) -> str:
    logger.debug(f"Generating non-streaming response for input: {input_text}")
    llm = OpenAI(
        temperature=0.5, 
        openai_api_key=openai_api_key,
        streaming=False
    )
    try:
        response_content = llm.invoke(input_text)
    except Exception as e:
        logger.error(f"Error during non-streaming response generation: {e}")
        return "Error generating response."

    logger.debug(f"Generated non-streaming response: {response_content}")  # Debugging: log the final response
    return response_content

st.title("Simple Chat")

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
    # Comment out the streaming response and use non-streaming for testing
    # response = generate_response(prompt)
    response = generate_response_non_streaming(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
