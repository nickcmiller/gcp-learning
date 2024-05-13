import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
st.title("Simple Chat App")

input_message = st.text_input("Type your message:")

if st.button("Send"):
    st.write(f"User: {input_message}")

    # Add a simple AI response
    response = "AI: Hello, how can I assist you today?"
    st.write(response)