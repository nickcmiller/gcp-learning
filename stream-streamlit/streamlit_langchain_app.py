import os
import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
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

def generate_response(input_text: str, chat_history: list):
    handler = StreamHandler()
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.5,
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[handler]
    )
    
    # Convert chat history to list of BaseMessages
    messages = []
    for message in chat_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        elif message["role"] == "system":
            messages.append(SystemMessage(content=message["content"]))

    # Add the new user message
    messages.append(HumanMessage(content=input_text))

    try:
        response = llm.stream(messages)
    except Exception as e:
        logger.error("Error during response generation: %s", e, exc_info=True)
        yield "Error generating response."

    for token in response:
        yield token

st.title("Simple Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Display chat messages from history on app rerun, excluding system messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Set prompt text based on whether the first question has been asked
prompt_text = "Ask a follow-up question..." if len(st.session_state.messages) > 1 else "Ask me anything..."

# Accept user input
if prompt := st.chat_input(prompt_text):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    chat_container = st.empty()  # Create an empty container for streaming
    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(prompt, st.session_state.messages))  # Use st.write_stream for streaming
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
