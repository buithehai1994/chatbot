from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import requests
import re

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get VADER sentiment score
def get_vader_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound'], sentiment['pos'], sentiment['neu'], sentiment['neg']

# Function to preprocess text: remove unnecessary characters, normalize spaces, etc.
def preprocess_text(text):
    # Remove unnecessary punctuation and normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

# FastAPI server URL (part of your chatbot)
API_URL = "https://chatbot-api-ewoq.onrender.com/chat"

# Streamlit app title
st.title("Chatbot 🤖")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display the chat history
for user_msg, bot_msg, user_sentiment_info, bot_sentiment_info in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    if bot_msg is not None:
        with st.chat_message("assistant"):
            st.markdown(bot_msg, unsafe_allow_html=True)
        st.markdown(bot_sentiment_info, unsafe_allow_html=True)

# User input field
if user_input := st.chat_input("Type your message and press Enter to chat..."):
    # Preprocess the user input
    preprocessed_user_input = preprocess_text(user_input)

    # Perform sentiment analysis on the user's input using VADER
    user_compound, user_pos, user_neu, user_neg = get_vader_sentiment(preprocessed_user_input)

    # Custom threshold for sentiment classification for user input
    if user_compound >= 0.25:
        user_sentiment_label = "very_positive"
    elif user_compound >= 0.05:
        user_sentiment_label = "slightly_positive"
    elif user_compound > -0.05 and user_compound < 0.05:
        user_sentiment_label = "neutral"
    elif user_compound <= -0.25:
        user_sentiment_label = "very_negative"
    else:
        user_sentiment_label = "slightly_negative"

    # Prepare sentiment info for the user's input
    user_sentiment_info = f"<br><span style='color:{'green' if user_sentiment_label == 'very_positive' else 
                                                'lightgreen' if user_sentiment_label == 'slightly_positive' else 
                                                'grey' if user_sentiment_label == 'neutral' else
                                                'orange' if user_sentiment_label == 'slightly_negative' else 
                                                'red'}; font-weight: bold;'>User Sentiment: {user_sentiment_label.replace('_', ' ').capitalize()}, Score: {user_compound:.2f}</span>"


     # Display the bot's response with sentiment info
    with st.chat_message("user"):
        st.write(user_input)
        st.markdown(user_sentiment_info, unsafe_allow_html=True)

    # Add user message to chat history
    st.session_state.chat_history.append((user_input, None, user_sentiment_info, None))

    # Send the user's message to the chatbot API
    payload = {"question": user_input}
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            bot_response = response.json().get("response", "No response from the bot.")
        else:
            bot_response = f"Error: Received status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        bot_response = f"An error occurred while connecting to the server: {e}"

    # Preprocess the bot's response
    preprocessed_bot_response = preprocess_text(bot_response)

    # Perform sentiment analysis on the bot's response using VADER
    bot_compound, bot_pos, bot_neu, bot_neg = get_vader_sentiment(preprocessed_bot_response)

    # Custom threshold for sentiment classification for bot's response
    if bot_compound >= 0.25:
        bot_sentiment_label = "very_positive"
    elif bot_compound >= 0.05:
        bot_sentiment_label = "slightly_positive"
    elif bot_compound > -0.05 and bot_compound < 0.05:
        bot_sentiment_label = "neutral"
    elif bot_compound <= -0.25:
        bot_sentiment_label = "very_negative"
    else:
        bot_sentiment_label = "slightly_negative"

    # Prepare sentiment info for the bot's response
    bot_sentiment_info = f"<br><span style='color:{'green' if bot_sentiment_label == 'very_positive' else 
                                              'lightgreen' if bot_sentiment_label == 'slightly_positive' else 
                                              'grey' if bot_sentiment_label == 'neutral' else
                                              'orange' if bot_sentiment_label == 'slightly_negative' else 
                                              'red'}; font-weight: bold;'>Bot Sentiment: {bot_sentiment_label.replace('_', ' ').capitalize()}, Score: {bot_compound:.2f}</span>"

    # Prepare the bot message
    bot_message = f"<p>{bot_response}</p>"

    # Update chat history with sentiment info
    st.session_state.chat_history[-1] = (user_input, bot_message, user_sentiment_info, bot_sentiment_info)

    # Display the bot's response with sentiment info
    with st.chat_message("assistant"):
        st.markdown(bot_message, unsafe_allow_html=True)
        st.markdown(bot_sentiment_info, unsafe_allow_html=True)
