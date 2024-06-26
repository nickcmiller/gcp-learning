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
    """
    This class is designed to handle streaming responses from an AI model and buffer the tokens that are generated.
    The class provides methods to add tokens to the buffer, process the tokens in the buffer, and check if the buffer is full. 
    These methods can be overridden in a subclass to customize the behavior of the class.

    The purpose of this class is to manage the flow of data in a streaming context, where the AI model generates tokens one at a time instead of all at once. 
    This is particularly useful for applications that need to process or display the tokens as they are generated, such as a chatbot or a real-time text generation tool.

    The class provides a buffer to store the tokens that are generated. 
    The size of the buffer can be specified when the class is instantiated. 
    When the buffer is full, the class processes the tokens in the buffer and then clears the buffer to make room for new tokens.

    Args:
        buffer_size (int): The maximum size of the buffer.

    Attributes:
        buffer (list): A list to store the tokens.
        buffer_size (int): The maximum size of the buffer.

    """

    def __init__(self, buffer_size=5):
        self.buffer = []
        self.buffer_size = buffer_size

    async def handle_response(self, response):
            """
            Handles the response received from a stream.

            Args:
                response: The response received from the stream.

            Yields:
                str: The text yielded from the buffer when it reaches the buffer size or when the response ends.
            """
            async for token in response:
                self.on_llm_new_token(token)
                if len(self.buffer) >= self.buffer_size:
                    text = ''.join(self.buffer)
                    self.buffer = []
                    logger.info(f"Buffer full, yielding text: {text}")
                    yield text
            if self.buffer:
                text = ''.join(self.buffer)
                self.buffer = []
                yield text

    def on_llm_new_token(self, token: str, **kwargs):
        """
        Callback function called when a new token is received.

        Args:
            token (str): The new token.

        """
        if isinstance(token, str):
            self.buffer.append(token)

async def generate_response(input_text: str, chat_history: list) -> Generator[str, None, None]:
    """
    Generates a response using the ChatOpenAI model.

    Args:
        input_text (str): The user's input text.
        chat_history (list): The chat history containing previous messages.

    Yields:
        str: The generated response tokens.

    Returns:
        Generator[str, None, None]: A generator that yields the response tokens.
    """
    handler = StreamHandler(buffer_size=1)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.5,
        openai_api_key=openai_api_key,
        streaming=True,
        callbacks=[handler]
    )

    messages = []

    # Iterate over each message in the chat history and create a corresponding message object
    for message in chat_history:
        # HumanMessage is a class that represents a message sent by the user in the chat.
        # It takes the content of the message as an argument.
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        # AIMessage is a class that represents a message generated by the AI assistant in the chat.
        # It takes the content of the message as an argument.
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
        # SystemMessage is a class that represents a system-generated message in the chat.
        # These messages are typically used for instructions or notifications.
        # It takes the content of the message as an argument.
        elif message["role"] == "system":
            messages.append(SystemMessage(content=message["content"]))
    
    # After processing all messages in the chat history, append a new HumanMessage with the input text
    messages.append(HumanMessage(content=input_text))

    try:
        response = llm.astream(messages)
    except Exception as e:
        logger.error("Error during response generation: %s", e, exc_info=True)
        yield "Error generating response."

    async for token in handler.handle_response(response):
        yield token

async def generate_and_display_response(prompt: str, messages: list) -> str:
    """
    Generates and displays a response based on the given prompt and messages.

    Args:
        prompt (str): The prompt for generating the response.
        messages (list): A list of messages exchanged between the user and the assistant.

    Returns:
        str: The generated response.

    """

    # Create an empty Streamlit container to display the assistant's message
    assistant_message_container = st.empty()

    response = ""
    assistant_message = ""
    
    # Create a new container within the assistant_message_container
    with assistant_message_container.container():
        
        # Call the generate_response function with the user's prompt and the chat history
        # This function returns an asynchronous generator that yields the assistant's response one token at a time
        async_gen = generate_response(prompt, messages)
        
        # Iterate over the tokens generated by the async generator
        async for token in async_gen:
            # Add the current token to the full response
            response += token
            
            # Add the current token to the assistant's message
            assistant_message += token
            
            # Display the assistant's message in the Streamlit container
            # The message is updated with each new token, creating a typing effect
            assistant_message_container.chat_message("assistant").markdown(assistant_message)
    
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    return response

st.title("Simple Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Iterate over each message in the session state's messages
for message in st.session_state.messages:
    # Check if the role of the message is not 'system'
    # 'system' messages are typically instructions or notifications, and we don't want to display them in the chat
    if message["role"] != "system":
        # Create a new chat message with the role of the current message
        # The 'with' statement is used here to apply a context to the chat message
        # The context is the role of the message, which indicates who sent the message ('user' or 'assistant')
        with st.chat_message(message["role"]):
            # Display the content of the message in the chat using Markdown formatting
            st.markdown(message["content"])

if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "Ask me anything..." if len(st.session_state.messages) > 2 else "Ask a follow-up question..."

# Check if the user has entered a prompt in the chat input field.
# The ':=' operator is known as the 'walrus operator' and is used to assign values to variables as part of an expression.
# If the user has entered a prompt, the 'if' statement will evaluate to True and the prompt will be assigned to the 'prompt' variable.
if prompt := st.chat_input(st.session_state.current_prompt):
    
    # Append the user's message to the 'messages' list in the session state.
    # Each message is represented as a dictionary with 'role' and 'content' keys.
    # The 'role' key indicates who sent the message ('user' in this case) and the 'content' key contains the text of the message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Create a new chat message from the user with the text of the prompt.
    # The 'with' statement is used here to apply a context to the chat message.
    # In this case, the context is 'user', which indicates that the message is from the user.
    with st.chat_message("user"):
        # Display the user's message in the chat using Markdown formatting.
        st.markdown(prompt)

    # Run the 'generate_and_display_response' function to generate a response from the AI assistant and display it in the chat.
    # The 'asyncio.run' function is used to run the 'generate_and_display_response' function, which is an asynchronous function.
    # The 'generate_and_display_response' function takes the user's prompt and the chat history as arguments.
    asyncio.run(generate_and_display_response(prompt, st.session_state.messages))
