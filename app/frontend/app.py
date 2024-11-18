import streamlit as st
import requests

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/chat"

# Streamlit app title
st.title("Interactive Chat with Groq Bot 🤖")

# Initialize conversation history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List to store (user, bot) messages

# Display the chat history dynamically
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    if bot_msg is not None:
        with st.chat_message("assistant"):
            st.write(bot_msg)

# User input field
if user_input := st.chat_input("Type your message and press Enter to chat..."):
    # Immediately display the user message
    with st.chat_message("user"):
        st.write(user_input)
    # Add user message to chat history
    st.session_state.chat_history.append((user_input, None))

    # Send POST request to FastAPI
    payload = {"question": user_input}
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            bot_response = response.json().get("response", "No response from the bot.")
        else:
            bot_response = f"Error: Received status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        bot_response = f"An error occurred while connecting to the server: {e}"

    # Immediately display the bot's response
    with st.chat_message("assistant"):
        st.write(bot_response)

    # Add bot response to chat history
    st.session_state.chat_history[-1] = (user_input, bot_response)
