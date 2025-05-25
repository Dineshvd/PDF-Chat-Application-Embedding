# from fastapi import FastAPI
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import google.generativeai as genai
# import os
# import uvicorn

# load_dotenv()

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or restrict as needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# gen_model = genai.GenerativeModel("gemini-2.0-flash")

# class EmbedRequest(BaseModel):
#     texts: list[str]

# class AnswerRequest(BaseModel):
#     prompt: str

# @app.post("/embed")
# async def embed(request: EmbedRequest):
#     vectors = embedding_model.encode(request.texts).tolist()
#     return {"embeddings": vectors}

# @app.post("/answer")
# async def answer(req: AnswerRequest):
#     try:
#         # response = gen_model.generate_content(req.prompt)
#         response= gen_model.generate_content(contents=[req.prompt])
#         # Extract the generated text
#         answer_text = response.text if hasattr(response, "text") else ""

#         return {"answer": answer_text}
#     except Exception as e:
#         print("Error:", e)
#         return {"error": str(e)}


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load .env variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI
app = FastAPI()

# origins = [
#     "http://localhost:3000",
#     "http://localhost:3001",
#     "http://localhost:5173",  # Vite default
#     "http://127.0.0.1:3000",
#     "http://127.0.0.1:5173"
# ]

# Enable CORS


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Required if you use "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class EmbedRequest(BaseModel):
    texts: list[str]

class AnswerRequest(BaseModel):
    prompt: str

# Embedding endpoint using Gemini
@app.post("/embed")
async def embed(request: EmbedRequest):
    try:
        vectors = []
        for text in request.texts:
            response = genai.embed_content(
                model="models/embedding-001",  # or "models/text-embedding-004" if available
                content=text,
                task_type="retrieval_document"
            )
            vectors.append(response["embedding"])
        return {"embeddings": vectors}
    except Exception as e:
        print("Error in /embed:", e)
        return {"error": str(e)}

# Answer endpoint using Gemini
@app.post("/answer")
async def answer(req: AnswerRequest):
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(
            contents=[req.prompt]
        )
        answer_text = response.text if hasattr(response, "text") else ""
        return {"answer": answer_text}
    except Exception as e:
        print("Error in /answer:", e)
        return {"error": str(e)}



