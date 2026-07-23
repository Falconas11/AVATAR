from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AVATAR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Github Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "project":"AVATAR",
        "status":"running"
    }