import streamlit as st
import requests
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# FastAPI server URL (part of your chatbot)
API_URL = "https://chatbot-api-ewoq.onrender.com/chat"

# Streamlit app title
st.title("Chatbot 🤖")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to classify sentiment
def classify_sentiment(compound_score):
    if compound_score >= 0.25:
        return "very_positive", 'green'
    elif compound_score >= 0.05:
        return "slightly_positive", 'lightgreen'
    elif compound_score > -0.05 and compound_score < 0.05:
        return "neutral", 'grey'
    elif compound_score <= -0.25:
        return "very_negative", 'red'
    else:
        return "slightly_negative", 'orange'

# Function to get diverse bot tone based on sentiment and alternating responses
def get_bot_tone(user_sentiment_label):
    # Response options for each sentiment
    positive_responses = [
        "I'm so glad to hear that! 😊 How can I assist you further?",
        "That's awesome! 🎉 Let me know if you need anything else.",
        "Yay! I'm happy you're feeling great! 😄 What can I do for you today?",
        "It’s wonderful to see you in such high spirits! 🌟 What can I help you with?",
        "Awesome! Let’s keep the good vibes going! How can I assist you today?",
        "This is great! Let’s make your day even better! What’s next?",
        "Wow, that's amazing! Keep it up, and let me know if you need anything!",
        "Such a great mood! 🎉 Let me know what you'd like to do next!"
    ]
    
    slightly_positive_responses = [
        "Great to know you're feeling good! How can I help you today?",
        "Awesome! 😊 Is there anything else you'd like to talk about?",
        "I'm glad to hear that! How can I assist you further?",
        "That's nice to hear! How can I be of service today?",
        "I’m happy things are looking up for you! What can I do to assist?",
        "I’m glad you're in a good mood! Let me know if you need anything.",
        "Nice to see you feeling positive! What would you like to do next?",
        "I'm so happy you're feeling this way! Let me know if you need help with anything."
    ]
    
    neutral_responses = [
        "Okay, let's continue. What would you like to do next?",
        "Got it. What can I help you with?",
        "Understood! Let me know what you'd like to do next.",
        "Alright! What's next on your mind?",
        "Sure, I’m ready to assist. What would you like to discuss?",
        "Got it! How can I assist you today?",
        "Let's keep going! What would you like to focus on?",
        "Sounds good. What's next?"
    ]
    
    slightly_negative_responses = [
        "I'm sorry to hear that. How can I help improve your experience?",
        "Oh no, I'm sorry you're feeling that way. Let me know how I can assist.",
        "I understand. What can I do to help make things better?",
        "I'm here for you! What can I do to help you feel better?",
        "I’m really sorry you're not feeling great. How can I assist you?",
        "I know things might be tough. Let me know how I can make it easier.",
        "I understand it's not the best situation. Let me know how I can help!",
        "I’m here to help, just let me know how I can make things better."
    ]
    
    very_negative_responses = [
        "Oh no, it seems like you're upset. Let me know how I can assist you.",
        "I'm really sorry to hear that. How can I make things right?",
        "I understand you're upset. How can I assist you to improve the situation?",
        "I’m here to help you. How can I make this better for you?",
        "I’m truly sorry things aren’t going well. Let me know what I can do.",
        "It sounds like you're going through a tough time. How can I support you?",
        "I'm really sorry to hear you're upset. Let me know what I can do to help.",
        "I understand you're having a rough time. Let's see how I can assist you."
    ]
    
    # Define a way to store the index for each sentiment type in session_state to ensure alternating responses
    if 'response_index' not in st.session_state:
        st.session_state['response_index'] = {
            'very_positive': 0,
            'slightly_positive': 0,
            'neutral': 0,
            'slightly_negative': 0,
            'very_negative': 0
        }

    # Set the response list based on the user's sentiment
    if user_sentiment_label == "very_positive":
        response_list = positive_responses
    elif user_sentiment_label == "slightly_positive":
        response_list = slightly_positive_responses
    elif user_sentiment_label == "neutral":
        response_list = neutral_responses
    elif user_sentiment_label == "slightly_negative":
        response_list = slightly_negative_responses
    elif user_sentiment_label == "very_negative":
        response_list = very_negative_responses
    else:
        response_list = []  # In case of an unknown sentiment label
    
    # Get the current response index for this sentiment label
    current_index = st.session_state['response_index'][user_sentiment_label]
    
    # Choose the response based on the current index, then update the index
    new_bot_response = response_list[current_index]
    
    # Update the index for the next response, looping back to 0 if the end of the list is reached
    next_index = (current_index + 1) % len(response_list)
    st.session_state['response_index'][user_sentiment_label] = next_index
    
    return new_bot_response

# Display the chat history with sentiment classification
for user_msg, bot_msg in st.session_state.chat_history:
    if user_msg:
        # Analyze user sentiment
        user_sentiment_score = analyzer.polarity_scores(user_msg)["compound"]
        user_sentiment_label, user_color = classify_sentiment(user_sentiment_score)

        # Display user message with sentiment label
        user_sentiment_info = f"<br><span style='color:{user_color}; font-weight: bold;'>User Sentiment: {user_sentiment_label.replace('_', ' ').capitalize()}, Score: {user_sentiment_score:.2f}</span>"
        with st.chat_message("user"):
            st.write(user_msg)
            st.markdown(user_sentiment_info, unsafe_allow_html=True)

    if bot_msg:
        # Analyze bot sentiment
        bot_sentiment_score = analyzer.polarity_scores(bot_msg)["compound"]
        bot_sentiment_label, bot_color = classify_sentiment(bot_sentiment_score)

        # Display bot message with sentiment label
        bot_sentiment_info = f"<br><span style='color:{bot_color}; font-weight: bold;'>Bot Sentiment: {bot_sentiment_label.replace('_', ' ').capitalize()}, Score: {bot_sentiment_score:.2f}</span>"
        with st.chat_message("assistant"):
            st.markdown(bot_msg, unsafe_allow_html=True)
            st.markdown(bot_sentiment_info, unsafe_allow_html=True)

# User input field
if user_input := st.chat_input("Type your message and press Enter to chat..."):
    # Analyze user sentiment and get sentiment label
    user_sentiment_score = analyzer.polarity_scores(user_input)["compound"]
    user_sentiment_label, user_color = classify_sentiment(user_sentiment_score)

    # Add user message to chat history
    st.session_state.chat_history.append((user_input, None))  # User message with None for bot message placeholder

    # Send the user's message to the chatbot API
    payload = {"question": user_input}
    bot_tone = ""  # Initialize bot_tone to avoid reference before assignment in case of errors
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            try:
                # Extract the message from the 'response' key in the API response
                bot_message = response.json().get("response")  # Get the actual response message
            except ValueError as e:
                # Handle errors if the response is not valid JSON
                st.error(f"Error decoding JSON response from API. Response: {response.text}")
                bot_message = "Oops! Something went wrong. Please try again later."
        else:
            # Handle unsuccessful API response
            st.error(f"API request failed with status code {response.status_code}.")
            bot_message = "There was an issue connecting to the chatbot. Please try again later."
    except requests.exceptions.RequestException as e:
        bot_message = f"An error occurred while connecting to the server: {e}"

    # Adjust the bot's tone based on user sentiment
    bot_tone = get_bot_tone(user_sentiment_label)  # Use the tone based on user sentiment

    # Modify the bot's response by appending the tone
    bot_message = f"{bot_tone} {bot_message}"

    # Add the bot's message with adjusted tone to chat history
    st.session_state.chat_history[-1] = (user_input, bot_message)

    # Display the entire chat history again, including the newly added user and bot messages
    for user_msg, bot_msg in st.session_state.chat_history:
        if user_msg:
            user_sentiment_score = analyzer.polarity_scores(user_msg)["compound"]
            user_sentiment_label, user_color = classify_sentiment(user_sentiment_score)
            user_sentiment_info = f"<br><span style='color:{user_color}; font-weight: bold;'>User Sentiment: {user_sentiment_label.replace('_', ' ').capitalize()}, Score: {user_sentiment_score:.2f}</span>"
            with st.chat_message("user"):
                st.write(user_msg)
                st.markdown(user_sentiment_info, unsafe_allow_html=True)

        if bot_msg:
            bot_sentiment_score = analyzer.polarity_scores(bot_msg)["compound"]
            bot_sentiment_label, bot_color = classify_sentiment(bot_sentiment_score)
            bot_sentiment_info = f"<br><span style='color:{bot_color}; font-weight: bold;'>Bot Sentiment: {bot_sentiment_label.replace('_', ' ').capitalize()}, Score: {bot_sentiment_score:.2f}</span>"
            with st.chat_message("assistant"):
                st.markdown(bot_msg, unsafe_allow_html=True)
                st.markdown(bot_sentiment_info, unsafe_allow_html=True)
