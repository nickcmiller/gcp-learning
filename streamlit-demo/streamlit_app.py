import os
from groq import Groq
import streamlit as st
import json

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Function to get response from LLama3 via Groq
def get_groq_response(messages):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to initialize chat history
def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

# Function to update chat history
def update_chat_history(user_message, ai_message):
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    st.session_state.chat_history.append({"role": "assistant", "content": ai_message})

# Initialize chat history
initialize_chat_history()

st.title("Groq Chat App")

# Display chat history in the correct order (from oldest to newest)
st.markdown("<div style='max-height: 400px; overflow-y: auto; padding: 10px;'>", unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right;'>You: {message['content']}</div>", unsafe_allow_html=True)
    elif message['role'] == 'assistant':
        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: left;'>AI: {message['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

input_message = st.text_input("Type your message:", key="input")

def send_message():
    if st.session_state.input:
        ai_message = get_groq_response(st.session_state.chat_history + [{"role": "user", "content": st.session_state.input}])
        if ai_message:
            update_chat_history(st.session_state.input, ai_message)
            st.session_state.input = ""  # Clear the input field
            st.experimental_rerun()  # Refresh the app to display the new message
    else:
        st.warning("Please type a message before sending.")

st.button("Send", on_click=send_message, key="send_button")

# Convert chat history to JSON
chat_history_json = json.dumps(st.session_state.chat_history)

# Enhanced CSS for better styling
st.markdown("""
    <style>
    .stTextInput div div input {
        padding: 10px !important;
        border: 2px solid #ccc !important;
        border-radius: 5px !important;
        outline: none !important;
        box-shadow: none !important;  /* Remove any box shadow which might be causing the red outline */
    }
    .stTextInput div div input:focus {
        outline: 2px solid blue !important;  /* Change outline color to blue when focused */
        border-color: blue !important;  /* Ensure border color is blue when focused */
        box-shadow: none !important;  /* Remove any box shadow which might be causing the red outline */
    }
    /* Override validation styles */
    .stTextInput div div input:invalid {
        border: 2px solid blue !important;
        box-shadow: none !important;
    }
    .stTextInput div div input:valid {
        border: 2px solid blue !important;
        box-shadow: none !important;
    }
    .stButton button {
        background-color: #ff6f61 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        cursor: pointer !important;
    }
    .stButton button:hover {
        background-color: #ff3b2f !important;
    }
    </style>
""", unsafe_allow_html=True)

# Improved JavaScript to trigger the send button on CMD + Enter
st.markdown("""
    <script>
    document.addEventListener("keydown", function(event) {
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            const sendButton = document.querySelector('button[aria-label="Send"]');
            if (sendButton) {
                sendButton.click();
            }
        }
    });
    </script>
""", unsafe_allow_html=True)
