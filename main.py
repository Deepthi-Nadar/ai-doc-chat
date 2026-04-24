from fastapi import FastAPI, UploadFile, File
from utils import extract_text, split_text, create_index, search
from openai import OpenAI


app = FastAPI()
openai = OpenAI()

index=None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global index
    text = extract_text(file.file)
    chunks = split_text(text)
    index = create_index(chunks)

    return {"message": "PDF processed successfully."}

@app.ask("/ask")
def ask_question(q: str):

    global index

    if index is None:
        return {"error": "please upload a PDF first."}

    results = search(index, q)
    context = " ".join(results)

    prompt = f""" Answer the question based on the context below. 
    Context: {context}
    Question: {q}
    """



    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content
    return {"answer": answer}



