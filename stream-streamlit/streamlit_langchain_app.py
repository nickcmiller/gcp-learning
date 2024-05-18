import os
import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Retrieve the API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OpenAI API Key is not set. Please set the API key in the environment variables.")

# Custom callback handler to capture streamed responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        if isinstance(token, str):
            self.text += token
            yield self.text

    def on_llm_end(self, response, **kwargs):
        return self.text

def generate_response(input_text: str):
    logger.debug(f"Generating response for input: {input_text}")
    handler = StreamHandler()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[handler]
    )
    try:
        logger.debug("Invoking LLM...")
        response = llm.stream(input_text)  # Use stream instead of invoke
        logger.debug("Invocation complete.")
        logger.debug(f"Raw response: {response}")  # Log the raw response
    except Exception as e:
        logger.error(f"Error during response generation: {e}", exc_info=True)
        yield "Error generating response."

    for token in response:
        yield token

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
    chat_container = st.empty()  # Create an empty container for streaming
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt))  # Use st.write_stream for streaming
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})