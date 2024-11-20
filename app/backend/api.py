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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Define the input structure for the question
class UserQuery(BaseModel):
    question: str

# Fetch Groq API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the Groq API key is missing
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to the .env file.")

model = 'gemma2-9b-it'

# Initialize the Groq chat model
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

# System prompt and memory for the conversation
system_prompt = 'You are a friendly conversational chatbot'
conversational_memory_length = 5  # Number of previous messages to remember

memory = ConversationBufferWindowMemory(
    k=conversational_memory_length, memory_key="chat_history", return_messages=True
)

@app.post("/chat")
async def chat_with_bot(user_query: UserQuery):
    """
    This endpoint receives a question from the user, processes it, and returns the chatbot's response.
    """
    user_question = user_query.question

    # Ensure the user question is not empty
    if not user_question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

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

    try:
        # Generate the chatbot's response
        response = conversation.predict(human_input=user_question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return {"response": response}

@app.get("/")
async def root():
    return {"message": "Welcome to the Groq Chatbot API. Use the /chat endpoint to chat."}
