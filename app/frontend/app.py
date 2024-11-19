import streamlit as st
import requests
from transformers import pipeline

# FastAPI server URL
API_URL = "https://chatbot-fvcf.onrender.com/chat"

# Load the sentiment analysis pipeline using Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

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

    # Perform sentiment analysis on the bot's response using Hugging Face pipeline
    sentiment_result = sentiment_analyzer(bot_response)
    sentiment_label = sentiment_result[0]['label']
    sentiment_score = sentiment_result[0]['score']

    # Append sentiment information to bot response
    sentiment_info = f"(Sentiment: {sentiment_label}, Confidence: {sentiment_score*100:.2f}%)"
    bot_response_with_sentiment = f"{bot_response} {sentiment_info}"

    # Immediately display the bot's response with sentiment
    with st.chat_message("assistant"):
        st.write(bot_response_with_sentiment)

    # Add bot response with sentiment to chat history
    st.session_state.chat_history[-1] = (user_input, bot_response_with_sentiment)
