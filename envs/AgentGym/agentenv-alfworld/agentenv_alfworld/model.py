from pydantic import BaseModel


class StepRequestBody(BaseModel):
    id: int
    action: str


class CloseRequestBody(BaseModel):
    id: int


class ResetRequestBody(BaseModel):
    id: int
    game: int
    world_type: str
