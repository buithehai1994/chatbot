import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Initialize FastAPI
app = FastAPI()

# Define the input structure for the question
class UserQuery(BaseModel):
    question: str

# Function to load the API key from the file in /etc/secrets
def load_groq_api_key():
    try:
        with open("/etc/secrets/api_key", "r") as f:
            return f.read().strip()  # Remove any trailing newlines or spaces
    except FileNotFoundError:
        raise ValueError("API key file not found.")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the API key: {e}")

# Load the API key from the secret file
groq_api_key = load_groq_api_key()

# Check if the Groq API key is missing
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

model = 'llama3-8b-8192'

# Initialize the Groq chat model
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

# System prompt and memory for the conversation
system_prompt = 'You are a friendly conversational chatbot'
conversational_memory_length = 5  # Number of previous messages to remember

memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

@app.post("/chat")
async def chat_with_bot(user_query: UserQuery):
    """
    This endpoint receives a question from the user, processes it, and returns the chatbot's response.
    """
    user_question = user_query.question

    if not user_question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    # Construct the chat prompt using various components
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),  # Persistent system prompt
            MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
            HumanMessagePromptTemplate.from_template("{human_input}"),  # User's current input
        ]
    )

    # Create a conversation chain
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,  # Use memory to keep track of chat history
    )

    # Generate the chatbot's response
    response = conversation.predict(human_input=user_question)
    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to the Groq Chatbot API. Use the /chat endpoint to chat."}
