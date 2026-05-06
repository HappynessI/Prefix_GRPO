from fastapi import FastAPI
import os
from .model import *
from .environment import server

app = FastAPI()

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    print("Running in VISUAL mode")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
def hello():
    return "This is environment BabyAI."


@app.post("/create")
async def create():
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.data_idx)

@app.get("/observation")
def get_observation(id: int):
    print(f"Observing environment {id}")
    return server.observe(id)

@app.post("/close")
def reset(body: CloseRequestBody):
    print("body", body)
    return server.close(body.id)

@app.post("/render")
def render_endpoint(body: CloseRequestBody):
    try:
        result = server.render(body.id)
        return result
    except Exception as e:
        return {"error": str(e)}
