import os
import logging
import streamlit as st
import openai
import asyncio
from langchain.callbacks.base import BaseCallbackHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Retrieve the API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OpenAI API Key is not set. Please set the API key in the environment variables.")
else:
    openai.api_key = openai_api_key

# Custom callback handler to capture streamed responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.content = []
        self.tokens_received = 0

    def on_new_token(self, token: str):
        self.content.append(token)
        self.tokens_received += 1
        logger.debug(f"Received token {self.tokens_received}: {token}")  # Debugging: log each token received

    def get_content(self):
        return ''.join(self.content)

async def generate_response(input_text: str) -> str:
    logger.debug(f"Generating response for input: {input_text}")
    handler = StreamHandler()

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",  # Ensure the correct model is used
            messages=[{"role": "user", "content": input_text}],
            max_tokens=256,
            temperature=0.5,
            stream=True  # Enable streaming
        )

        logger.debug("Invocation complete.")
        async for event in response:
            if 'choices' in event and len(event['choices']) > 0:
                choice = event['choices'][0]
                if 'delta' in choice and 'content' in choice['delta']:
                    handler.on_new_token(choice['delta']['content'])
        
    except Exception as e:
        logger.error(f"Error during response generation: {e}", exc_info=True)
        return "Error generating response."

    response_content = handler.get_content()
    logger.debug(f"Generated response: {response_content}")  # Debugging: log the final response
    return response_content

def generate_response_sync(input_text: str) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(generate_response(input_text))

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
    response = generate_response_sync(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
