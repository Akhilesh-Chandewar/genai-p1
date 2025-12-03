from fastapi import FastAPI
from pydantic import BaseModel
from ollama import Client

app = FastAPI()
client = Client(
    host="http://localhost:11434",
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/contact-us")
def read_contact():  # Changed function name to avoid duplicate
    return {
        "email": "akhil.chandewar00@gmail.com"
    }  # Fixed typo: gmail,com -> gmail.com


@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat(
        model="gemma:2b", messages=[{"role": "user", "content": request.message}]
    )

    return {"response": response.message.content}
