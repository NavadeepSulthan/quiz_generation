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
        # Extract JSON safely
        json_start = response.text.find("{")
        json_end = response.text.rfind("}")
        json_text = response.text[json_start:json_end+1]  # Extract JSON part

        quiz_json = json.loads(json_text)
    except Exception as e:
        quiz_json = {"error": f"Invalid JSON response from API: {str(e)}"}
    
    return quiz_json

# API Endpoint
@app.route('/generate_quiz', methods=['POST'])
def quiz_endpoint():
    data = request.get_json()
    summary_text = data.get("summary", "").strip()

    if not summary_text:
        return jsonify({"error": "Summary text is required"}), 400

    quiz = generate_quiz(summary_text)
    return jsonify(quiz)

# Run Flask server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
