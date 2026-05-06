import os
import threading
from collections import Counter

from fastapi import FastAPI

from .model import *
from .env_wrapper import server

app = FastAPI()
LOG_EVERY_N = 20
_request_log_counts = Counter()
_request_log_lock = threading.Lock()


def _should_log(name: str, every: int = LOG_EVERY_N) -> bool:
    with _request_log_lock:
        _request_log_counts[name] += 1
        count = _request_log_counts[name]
    return count == 1 or count % every == 0

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
    return "This is environment TextCraft."


@app.post("/create")
async def create(body: CreateRequestBody):
    return server.create(body.commands, body.goal, body.data_idx)


@app.post("/step")
async def step(body: StepRequestBody):
    if _should_log("step"):
        print(f"/step [{_request_log_counts['step']}] {body.id} {body.action}")
    return server.step(body.id, body.action)


@app.post("/reset")
async def reset(body: ResetRequestBody):
    if _should_log("reset"):
        print(f"/reset [{_request_log_counts['reset']}] {body.id} {body.data_idx}")
    return server.reset(body.id, body.data_idx)


@app.get("/observation")
async def get_observation(id: int):
    if _should_log("observation"):
        print(f"/observation [{_request_log_counts['observation']}] {id}")
    return server.get_observation(id)


@app.get("/commands")
async def get_commands(id: int):
    return server.get_commands(id)


@app.get("/goal")
async def get_goal(id: int):
    return server.get_goal(id)


@app.get("/detail")
async def get_detailed_info(id: int):
    return server.get_detailed_info(id)

@app.post("/close")
async def close(body: CloseRequestBody):
    if _should_log("close"):
        print(f"/close [{_request_log_counts['close']}] {body.id}")
    return server.close(body.id)
