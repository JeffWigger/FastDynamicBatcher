import time

from contextlib import asynccontextmanager

import pytest

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI
from pydantic import BaseModel

from fast_dynamic_batcher.dyn_batcher import DynBatcher
from fast_dynamic_batcher.inference_template import InferenceModel
from fast_dynamic_batcher.models import Task


class TestModel(InferenceModel):
    def __init__(self):
        super().__init__()

    def infer(self, tasks: list[Task]) -> list[Task]:
        time.sleep(0.05)
        return tasks


class Input(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    RunVar("_default_thread_limiter").set(CapacityLimiter(16))
    global dyn_batcher
    dyn_batcher = DynBatcher(TestModel, 8)
    time.sleep(3)
    yield
    print("FastAPI shutdown")
    dyn_batcher.stop()


@pytest.fixture
def wrapped_app():
    # dyn_batcher = DynBatcher(TestModel)
    app = FastAPI(lifespan=lifespan)

    @app.post("/predict")
    async def predict(input: Input):
        res = await dyn_batcher.process_batched(input.model_dump_json())
        return res

    yield app
    # dyn_batcher.stop()
