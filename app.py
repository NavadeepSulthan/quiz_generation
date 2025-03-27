import random
import smtplib
from email.message import EmailMessage
from reportlab.pdfgen import canvas
import requests
import nltk
from nltk.tokenize import sent_tokenize
import re
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Hugging Face API for Question Generation
HF_API_URL = "https://api-inference.huggingface.co/models/ramsrigouthamg/t5_squad_v1"
HF_HEADERS = {"Authorization": "Bearer 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"}

# Function to generate questions using Hugging Face API
def generate_question(context, answer):
    payload = {"inputs": f"generate question: {answer} context: {context}"}
    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    return "Question generation failed"

# Extract key phrases (answers) from text
def extract_key_phrases(text):
    sentences = sent_tokenize(text)
    key_phrases = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if len(words) > 3:
            key_phrases.append(random.choice(words))
    return key_phrases[:5]

# Generate MCQs and Fill-in-the-Blanks
def generate_quiz(text):
    mcqs = []
    fill_in_the_blanks = []
    key_phrases = extract_key_phrases(text)
    for phrase in key_phrases:
        question = generate_question(text, phrase)
        incorrect_options = random.sample([kp for kp in key_phrases if kp != phrase], min(3, len(key_phrases)-1))
        options = [phrase] + incorrect_options
        random.shuffle(options)
        mcqs.append({"question": question, "options": options, "answer": phrase})
        blank_sentence = re.sub(rf"\b{phrase}\b", "", text, count=1)
        fill_in_the_blanks.append({"question": blank_sentence, "answer": phrase})
    return mcqs, fill_in_the_blanks

# API Endpoint to get Quiz
def fetch_summarized_text():
    # Assuming the summarized text is stored in SummaryScreen
    summarized_text = ""  # Fetch this from SummaryScreen
    return summarized_text

@app.get("/generate-quiz")
def get_quiz():
    summarized_text = fetch_summarized_text()
    if not summarized_text:
        raise HTTPException(status_code=400, detail="No summarized text available")
    mcqs, fibs = generate_quiz(summarized_text)
    return {"mcqs": mcqs, "fill_in_the_blanks": fibs}

# Function to generate quiz results PDF
def generate_pdf(student_name, score, total_questions):
    pdf_filename = f"{student_name}_Quiz_Result.pdf"
    c = canvas.Canvas(pdf_filename)
    c.setFont("Helvetica", 16)
    c.drawString(100, 750, "Quiz Result Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Student Name: {student_name}")
    c.drawString(100, 700, f"Score: {score} / {total_questions}")
    c.drawString(100, 680, f"Performance: {'Excellent' if score >= total_questions * 0.8 else 'Needs Improvement'}")
    c.save()
    return pdf_filename

# Function to send email with quiz results
def send_email(student_name, parent_email, pdf_filename):
    sender_email = "udaybabupeeka45@gmail.com"
    sender_password = "mxmd vagx kmgj ijag"
    
    msg = EmailMessage()
    msg["Subject"] = "Your Child's Quiz Report"
    msg["From"] = sender_email
    msg["To"] = parent_email
    msg.set_content(f"Dear Parent,\n\nAttached is the quiz report for {student_name}.\n\nBest Regards,\nQuiz Team")
    
    with open(pdf_filename, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=pdf_filename)
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
