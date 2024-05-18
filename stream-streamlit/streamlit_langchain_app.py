import os
import logging
import asyncio
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

class StreamHandler(BaseCallbackHandler):
    def __init__(self, buffer_size=1):  # Reduced buffer size for more prompt updates
        self.buffer = []
        self.buffer_size = buffer_size

    async def handle_response(self, response):
        async for token in response:
            self.on_llm_new_token(token)
            if len(self.buffer) >= self.buffer_size:
                text = ''.join(self.buffer)
                self.buffer = []
                yield text
        if self.buffer:
            text = ''.join(self.buffer)
            self.buffer = []
            yield text

    def on_llm_new_token(self, token: str, **kwargs):
        if isinstance(token, str):
            self.buffer.append(token)

async def generate_response(input_text: str, chat_history: list):
    handler = StreamHandler(buffer_size=1)  # Reduced buffer size
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
        response = llm.astream(messages)
    except Exception as e:
        logger.error("Error during response generation: %s", e, exc_info=True)
        yield "Error generating response."

    async for token in handler.handle_response(response):
        yield token

def generate_response_sync(input_text: str, chat_history: list):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = []
    async_gen = generate_response(input_text, chat_history)
    while True:
        try:
            token = loop.run_until_complete(async_gen.__anext__())
            result.append(token)
        except StopAsyncIteration:
            break
    return ''.join(result)

st.title("Simple Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Display chat messages from history on app rerun, excluding system messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "Ask me anything..." if len(st.session_state.messages) > 2 else "Ask a follow-up question..."

# Accept user input
if prompt := st.chat_input(st.session_state.current_prompt):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    response = ""
    chat_container = st.empty()
    with st.chat_message("assistant"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = generate_response(prompt, st.session_state.messages)
        while True:
            try:
                token = loop.run_until_complete(async_gen.__anext__())
                response += token
                chat_container.markdown(response)
            except StopAsyncIteration:
                break
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
