from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import os
import uvicorn

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
gen_model = genai.GenerativeModel("gemini-2.0-flash")

class EmbedRequest(BaseModel):
    texts: list[str]

class AnswerRequest(BaseModel):
    prompt: str

@app.post("/embed")
async def embed(request: EmbedRequest):
    vectors = embedding_model.encode(request.texts).tolist()
    return {"embeddings": vectors}

@app.post("/answer")
async def answer(req: AnswerRequest):
    try:
        # response = gen_model.generate_content(req.prompt)
        response= gen_model.generate_content(contents=[req.prompt])
        # Extract the generated text
        answer_text = response.text if hasattr(response, "text") else ""

        return {"answer": answer_text}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}


