from fastapi import FastAPI, UploadFile, File
from utils import extract_text, split_text, create_index, search
from openai import OpenAI

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

from fastapi import Query


from fastapi import FastAPI, Query

from fastapi.responses import StreamingResponse


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

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        return {
            response.output[0].content[0].text
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