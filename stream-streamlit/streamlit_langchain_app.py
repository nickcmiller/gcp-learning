import os
import logging
import asyncio
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OpenAI API Key is not set. Please set the API key in the environment variables.")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, buffer_size=5):
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
    handler = StreamHandler(buffer_size=5)  # Increase buffer size for batch updates
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[handler]
    )

    messages = []
    for message in chat_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        elif message["role"] == "system":
            messages.append(SystemMessage(content=message["content"]))

    messages.append(HumanMessage(content=input_text))

    try:
        response = llm.astream(messages)
    except Exception as e:
        logger.error("Error during response generation: %s", e, exc_info=True)
        yield "Error generating response."

    async for token in handler.handle_response(response):
        yield token

async def generate_and_display_response(prompt, messages):
    response = ""
    assistant_message_container = st.empty()
    assistant_message = ""

    with assistant_message_container.container():
        async_gen = generate_response(prompt, messages)
        async for token in async_gen:
            response += token
            assistant_message += token
            # Use st.chat_message to keep the format consistent
            assistant_message_container.chat_message("assistant").markdown(assistant_message)
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    return response

st.title("Simple Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "Ask me anything..." if len(st.session_state.messages) > 2 else "Ask a follow-up question..."

if prompt := st.chat_input(st.session_state.current_prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    asyncio.run(generate_and_display_response(prompt, st.session_state.messages))
