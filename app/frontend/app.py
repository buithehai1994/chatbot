import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# FastAPI server URL
API_URL = "https://chatbot-fvcf.onrender.com/chat"

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit app title
st.title("Interactive Chat with Groq Bot 🤖")

# Initialize conversation history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List to store (user, bot, sentiment) messages

# Display the chat history dynamically
for user_msg, bot_msg, sentiment_info in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    if bot_msg is not None:
        with st.chat_message("assistant"):
            st.markdown(bot_msg, unsafe_allow_html=True)
            st.markdown(sentiment_info, unsafe_allow_html=True)

# User input field
if user_input := st.chat_input("Type your message and press Enter to chat..."):
    # Immediately display the user message
    with st.chat_message("user"):
        st.write(user_input)

    # Add user message to chat history
    st.session_state.chat_history.append((user_input, None, None))

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

    # Perform sentiment analysis using VADER
    sentiment_score = analyzer.polarity_scores(bot_response)

    # Determine sentiment label and color based on compound score
    if sentiment_score['compound'] >= 0.05:
        sentiment_label = "POSITIVE"
        sentiment_color = "green"
    elif sentiment_score['compound'] <= -0.05:
        sentiment_label = "NEGATIVE"
        sentiment_color = "red"
    else:
        sentiment_label = "NEUTRAL"
        sentiment_color = "gray"

    # Create sentiment info string with color
    sentiment_info = f"<br><span style='color:{sentiment_color}; font-weight: bold;'>Sentiment: {sentiment_label}, Confidence: {sentiment_score['compound']*100:.2f}%</span>"

    # Prepare the bot message with sentiment
    bot_message = f"<p>{bot_response}</p>"

    # Add the bot's message and sentiment to the history
    st.session_state.chat_history[-1] = (user_input, bot_message, sentiment_info)

    # Immediately display the bot's response with formatting
    with st.chat_message("assistant"):
        st.markdown(bot_message, unsafe_allow_html=True)
        st.markdown(sentiment_info, unsafe_allow_html=True)
