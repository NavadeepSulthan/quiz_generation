from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import random
import nltk
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize

# ✅ Set NLTK Data Path (Fixes Render Issue)
NLTK_DATA_PATH = "/opt/render/nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# ✅ Ensure all required NLTK resources are downloaded
nltk.download('punkt', download_dir=NLTK_DATA_PATH)
nltk.download('averaged_perceptron_tagger', download_dir=NLTK_DATA_PATH)
nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
nltk.download('omw-1.4', download_dir=NLTK_DATA_PATH)

# Hugging Face API Setup
HF_API_KEY = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/ramsrigouthamg/t5_squad_v1"

# Initialize FastAPI app
app = FastAPI()

# Define Input Model
class QuizInput(BaseModel):
    text: str

# Extract key phrases from text
def extract_key_phrases(text):
    sentences = sent_tokenize(text)
    key_phrases = set()

    for sentence in sentences:
        words = word_tokenize(sentence)
        key_phrases.update([word for word in words if len(word) > 3 and word.isalpha()])

    return list(key_phrases)

# Generate questions using Hugging Face API
def generate_question(context, answer):
    input_text = f"generate question: {answer} context: {context}"
    payload = {"inputs": input_text}
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        response = requests.post(HF_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()[0]['generated_text']
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error from Hugging Face API: {str(e)}")

# Generate MCQs and Fill-in-the-Blanks
def generate_quiz(text):
    mcqs = []
    fill_in_the_blanks = []
    key_phrases = extract_key_phrases(text)

    if not key_phrases:
        raise HTTPException(status_code=400, detail="No valid key phrases found in the text.")

    for phrase in key_phrases:
        try:
            question = generate_question(text, phrase)
            incorrect_options = random.sample([kp for kp in key_phrases if kp != phrase], min(3, len(key_phrases)-1))
            options = [phrase] + incorrect_options
            random.shuffle(options)
            mcqs.append({"question": question, "options": options, "answer": phrase})

            blank_sentence = re.sub(rf"\b{phrase}\b", "_____", text, count=1)
            fill_in_the_blanks.append({"question": blank_sentence, "answer": phrase})
        except Exception as e:
            print(f"Skipping phrase '{phrase}' due to error: {e}")

    return mcqs, fill_in_the_blanks

# Define the API endpoint for quiz generation
@app.post("/generate_quiz")
async def create_quiz(quiz_input: QuizInput):
    try:
        mcqs, fill_in_the_blanks = generate_quiz(quiz_input.text)
        return {"mcqs": mcqs, "fill_in_the_blanks": fill_in_the_blanks}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
