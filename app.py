from flask import Flask, request, jsonify
import google.generativeai as genai
import json

# Configure Google API Key
GOOGLE_API_KEY = "AIzaSyCYbTBMymPADTspJCxlgWbjFY-5wthkBgA"
genai.configure(api_key=GOOGLE_API_KEY)

# Select Gemini model
model = genai.GenerativeModel("gemini-1.5-pro-002")

app = Flask(__name__)

# Function to generate quiz from summary
def generate_quiz(summary_text):
    prompt = f"""
    Generate 10 multiple-choice quiz questions from the given summary. Output only JSON format without explanations.
    
    Summary:
    {summary_text}
    
    JSON format:
    {{
      "quiz": [
        {{"question": "Your question here",
          "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
          "answer": "Correct option"
        }},
        ...
      ]
    }}
    """
    
    response = model.generate_content(prompt)

    try:
        # Extract JSON response
        json_text = response.text.strip().split("```json")[-1].split("```")[0].strip()
        quiz_json = json.loads(json_text)
    except json.JSONDecodeError:
        quiz_json = {"error": "Invalid JSON response from API"}
    
    return quiz_json

# API Endpoint
@app.route('/generate_quiz', methods=['POST'])
def quiz_endpoint():
    data = request.get_json()
    summary_text = data.get("summary", "")

    if not summary_text:
        return jsonify({"error": "Summary text is required"}), 400

    quiz = generate_quiz(summary_text)
    return jsonify(quiz)

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
