import os
import openai
import streamlit as st
import json

# Function to initialize chat history
def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

# Function to get AI response
def get_ai_response(messages):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message['content']
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

input_message = st.text_input("Type your message:")

if st.button("Send"):
    if input_message:
        ai_message = get_ai_response(st.session_state.chat_history + [{"role": "user", "content": input_message}])
        if ai_message:
            update_chat_history(input_message, ai_message)
            st.write(f"AI: {ai_message}")
    else:
        st.warning("Please type a message before sending.")

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.write(f"You: {message['content']}")
    elif message['role'] == 'assistant':
        st.write(f"AI: {message['content']}")

# Convert chat history to JSON
chat_history_json = json.dumps(st.session_state.chat_history)
