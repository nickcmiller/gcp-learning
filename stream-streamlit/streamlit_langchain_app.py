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
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)
def generate_response(input_text: str) -> str:
    logger.debug(f"Generating response for input: {input_text}")
    chat_box = st.empty()  # Create an empty container for streaming
    handler = StreamHandler(chat_box)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[handler]
    )
    try:
        logger.debug("Invoking LLM...")
        response = llm.invoke(input_text)  # Use invoke instead of __call__
        logger.debug("Invocation complete.")
        logger.debug(f"Raw response: {response}")  # Log the raw response
    except Exception as e:
        logger.error(f"Error during response generation: {e}", exc_info=True)
        return "Error generating response."

    response_content = handler.text
    logger.debug(f"Generated response: {response_content}")  # Debugging: log the final response
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
    response = generate_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})