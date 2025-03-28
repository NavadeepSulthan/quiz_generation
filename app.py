import google.generativeai as genai

# Set your API Key
GOOGLE_API_KEY = "AIzaSyCYbTBMymPADTspJCxlgWbjFY-5wthkBgA"
genai.configure(api_key=GOOGLE_API_KEY)
# Select the latest Gemini model
model = genai.GenerativeModel("gemini-1.5-pro-002")
import json


# Function to generate quiz
def generate_quiz(summary_text):
    prompt = f"""
    Generate 10 multiple-choice quiz from the given summary. Output only JSON format without explanations.
    
    Summary:
    {summary_text}
    
    JSON format:
    {{
      "quiz": [
        {{
          "question": "Your question here",
          "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
          "answer": "Correct option"
        }},
        ...
      ]
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        # Extract the valid JSON part
        json_text = response.text.strip().split("```json")[-1].split("```")[0].strip()
        quiz_json = json.loads(json_text)
    except json.JSONDecodeError:
        quiz_json = {"error": "Invalid JSON response from API"}
    
    return quiz_json

# Example summary
summary_text = "Machine learning is a subset of AI that enables systems to learn from data and make predictions."

# Generate quiz
quiz = generate_quiz(summary_text)

# Print structured output
print(json.dumps(quiz, indent=2))
