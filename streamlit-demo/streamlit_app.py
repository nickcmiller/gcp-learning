import os
from groq import Groq
import streamlit as st
import json

##################
# GROQ CLIENT SETUP
##################

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to get response from LLama3 via Groq
def get_groq_response(messages: list):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

####################
# APPLICATION LOGIC
####################

# Function to initialize chat history
def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

# Function to update chat history
def update_chat_history(user_message:str , ai_message:str):
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    st.session_state.chat_history.append({"role": "assistant", "content": ai_message})

# Function to clear the input field after sending a message
def clear_input():
    st.session_state["input"] = ""

# Function to send message upon button click or Enter key press
def send_message():
    if st.session_state.input:
        # Get response from Groq using the current input and chat history
        ai_message = get_groq_response(st.session_state.chat_history + [{"role": "user", "content": st.session_state.input}])
        if ai_message:
            # Add the input and Groq response to chat history
            update_chat_history(st.session_state.input, ai_message)
            # Clear the input field
            clear_input()

#######################
# STREAMLIT COMPONENTS
#######################

# Initialize chat history
initialize_chat_history()

# Set the title of the app
st.title("Groq Chat App")

# Display chat history in the correct order (from oldest to newest)
st.markdown("<div style='max-height: 400px; overflow-y: auto; padding: 10px;'>", unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        # Make the user messages blue and alligned to the right
        st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right; color: black; border: 4px solid #b0c7e1;'><b style='font-size: 16px;'>You</b><br> {message['content']}</div>", unsafe_allow_html=True)
    elif message['role'] == 'assistant':
        # Make the assistant messages gray and aligned to the left
        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: left; color: black; border: 4px solid #bcbcbc;'><b style='font-size: 16px;'>AI</b><br> {message['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Input field for user messages
input_message = st.text_input("Type your message:", key="input", on_change=send_message)

# Send button to trigger sending the message
st.button("Send", on_click=send_message, key="send_button")

# Convert chat history to JSON for data export
chat_history_json = json.dumps(st.session_state.chat_history)

####################
# STYLING WITH CSS
####################

# Changed colors and styles for the better
st.markdown("""
    <style>
    .stTextInput .st-c1, .stTextInput .st-c0, .stTextInput .st-bz, .stTextInput .st-by {
        border-color: #808080 !important;
    }
    .stButton button {
        background-color: #707070 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        cursor: pointer !important;
    }
    .stButton button:hover {
        background-color: #808080 !important;
    }
    </style>
""", unsafe_allow_html=True)
