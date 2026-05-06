from fastapi import FastAPI
from .env_wrapper import server
from .model import *

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment AlfWorld."


@app.post("/create")
def create():
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)


@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.id)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.game, body.world_type)


@app.get("/stats")
def stats():
    return server.stats()


@app.get("/available_actions")
def get_available_actions(id: int):
    return server.get_available_actions(id)


@app.get("/observation")
def get_observation(id: int):
    return server.get_observation(id)


@app.get("/detail")
def get_detailed_info(id: int):
    return server.get_detailed_info(id)
