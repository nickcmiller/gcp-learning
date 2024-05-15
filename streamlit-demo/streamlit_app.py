import os
from openai import OpenAI
import streamlit as st
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to initialize chat history
def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

# Function to get AI response
def get_ai_response(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to update chat history
def update_chat_history(user_message, ai_message):
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    st.session_state.chat_history.append({"role": "assistant", "content": ai_message})

# Initialize chat history
initialize_chat_history()

st.title("Simple Chat App")

# Display chat history in the correct order (from oldest to newest)
st.markdown("<div style='max-height: 400px; overflow-y: auto; padding: 10px;'>", unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.markdown(f"<div style='background-color: #e1f5fe; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: right;'>You: {message['content']}</div>", unsafe_allow_html=True)
    elif message['role'] == 'assistant':
        st.markdown(f"<div style='background-color: #fff3e0; padding: 10px; border-radius: 10px; margin-bottom: 10px; text-align: left;'>AI: {message['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

input_message = st.text_input("Type your message:", key="input")

if st.button("Send", key="send"):
    if input_message:
        ai_message = get_ai_response(st.session_state.chat_history + [{"role": "user", "content": input_message}])
        if ai_message:
            update_chat_history(input_message, ai_message)
            st.experimental_rerun()  # Refresh the app to display the new message
    else:
        st.warning("Please type a message before sending.")

# Convert chat history to JSON
chat_history_json = json.dumps(st.session_state.chat_history)

# CSS for better styling
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        padding: 10px;
        border: 2px solid #ccc;
        border-radius: 5px;
        outline-color: blue;  /* Change outline color to blue */
    }
    .stTextInput>div>div>input:focus {
        outline: 2px solid blue;  /* Change outline color to blue when focused */
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #ff3b2f;
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript to trigger the send button on CMD + Enter
st.markdown("""
    <script>
    document.addEventListener("keydown", function(event) {
        if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            document.querySelector('button[title="send"]').click();
        }
    });
    </script>
""", unsafe_allow_html=True)
