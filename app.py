from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import random
import nltk
from nltk.tokenize import sent_tokenize
import re
import os

# Set up Hugging Face API key (replace YOUR_HUGGINGFACE_API_KEY with your actual API key)
HF_API_KEY = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/ramsrigouthamg/t5_squad_v1"

# Initialize FastAPI app
app = FastAPI()

# Download necessary NLTK resources
nltk.download('punkt')

# Define the input format using Pydantic
class QuizInput(BaseModel):
    text: str

# Extract key phrases (answers) from text
def extract_key_phrases(text):
    sentences = sent_tokenize(text)
    key_phrases = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        # Extract meaningful phrases
        key_phrases.extend([word for word in words if len(word) > 3 and word.isalpha()])
    # Get distinct meaningful key phrases
    return list(set(key_phrases))

# Function to generate questions dynamically using Hugging Face API
def generate_question(context, answer):
    input_text = f"generate question: {answer} context: {context}"
    payload = {"inputs": input_text}
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    
    response = requests.post(HF_API_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error from Hugging Face API")
    
    question = response.json()[0]['generated_text']
    return question

# Generate MCQs and Fill-in-the-Blanks
def generate_quiz(text):
    mcqs = []
    fill_in_the_blanks = []
    key_phrases = extract_key_phrases(text)

    # Generate MCQs
    for phrase in key_phrases:
        question = generate_question(text, phrase)
        incorrect_options = random.sample([kp for kp in key_phrases if kp != phrase], min(3, len(key_phrases)-1))
        options = [phrase] + incorrect_options
        random.shuffle(options)
        mcqs.append({"question": question, "options": options, "answer": phrase})

        # Generate fill-in-the-blank
        blank_sentence = re.sub(rf"\b{phrase}\b", "_____", text, count=1)
        fill_in_the_blanks.append({"question": blank_sentence, "answer": phrase})

    return mcqs, fill_in_the_blanks

# Define the API endpoint for quiz generation
@app.post("/generate_quiz")
async def create_quiz(quiz_input: QuizInput):
    try:
        mcqs, fill_in_the_blanks = generate_quiz(quiz_input.text)

        # Return the result as a JSON object
        return {"mcqs": mcqs, "fill_in_the_blanks": fill_in_the_blanks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
