from fastapi import FastAPI, UploadFile, File, Query
from utils import extract_text, split_text, create_index, search
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
index=None
indexes = {}
documents_store = {}
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global indexes, documents_store

    text = extract_text(file.file)
    chunks = split_text(text)

    index = create_index(chunks)

    filename = file.filename

    indexes[filename] = index
    documents_store[filename] = chunks

    return {"message": f"{filename} uploaded successfully"}



@app.post("/ask")
async def ask(q: str = Query(...)):
    global indexes, documents_store

    try:
        all_results = []

        for filename, index in indexes.items():
            chunks = documents_store[filename]

            results = search(index, q, chunks)

            for r in results:
                all_results.append(f"[{filename}] {r}")

        # ⚠️ IMPORTANT: if no PDFs uploaded
        if not all_results:
            return {"Please upload a PDF first."}

        context = "\n".join(all_results[:5])

        # ✅ PASS CONTEXT HERE
        prompt = f"""
        Answer the question ONLY using the context below.

        Context:
        {context}

        Question:
        {q}
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return {
            "answer": response.text
        }

    except Exception as e:
        return {"error": str(e)}
    

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)