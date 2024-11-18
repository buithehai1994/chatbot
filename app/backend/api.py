from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

app = FastAPI()

# Load a generative model (e.g., GPT-2, GPT-Neo)
model_name = "EleutherAI/gpt-neo-1.3B"  # You can use other models like GPT-2, GPT-3, etc.
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Create the Hugging Face pipeline for text generation
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Wrap the Hugging Face pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define LangChain's ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
        ("assistant", "{output}"),
    ]
)

# Define the LangChain chain that uses the generative model
llm_chain = LLMChain(prompt=chat_prompt, llm=llm)

@app.post("/chat")
async def chat(message: str):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Use LangChain to generate a response
    prompt = f"User: {message}\nAssistant:"
    generated_response = llm_chain.run(input=prompt)

    return {"response": generated_response}
