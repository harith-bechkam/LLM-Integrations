import PyPDF2
import docx2txt
# from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif filename.endswith(".docx"):
        text = docx2txt.process(file)
        return text
    else:
        raise ValueError("Unsupported file format")


def calculate_similarity(resume_text, job_desc):
    # vectorizer = TfidfVectorizer(stop_words="english")
    # vectors = vectorizer.fit_transform([text1, text2])
    # similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    # return round(similarity * 100, 2)

    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0] * 100, 2)
