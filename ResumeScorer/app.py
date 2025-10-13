from flask import Flask, request, jsonify
from flask_cors import CORS

from helper import calculate_similarity, extract_text

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "flask app is running..."


@app.route('/score', methods=['POST'])
def score_resume():
    try:
        # 1️⃣ Check if file is provided
        if 'resume' not in request.files:
            raise ValueError("No resume file uploaded.")

        file = request.files['resume']

        # 2️⃣ Check file extension
        if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".docx")):
            raise ValueError("Unsupported file format. Please upload PDF or DOCX.")

        # 3️⃣ Check job description
        job_desc = request.form.get('jobDescription', '').strip()
        if not job_desc:
            raise ValueError("Job description cannot be empty.")

        # 4️⃣ Extract text from resume
        resume_text = extract_text(file)
        if not resume_text.strip():
            raise ValueError("Resume content is empty or unreadable.")

        # 5️⃣ Calculate similarity score
        score = calculate_similarity(resume_text, job_desc)
        score = round(float(score), 2)

        return jsonify({"score": score})

    except ValueError as ve:
        # Custom validation errors
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        # Unexpected errors
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
