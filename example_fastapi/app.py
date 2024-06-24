from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn

from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI, Form, UploadFile
from PIL import Image
from torchvision.models import ResNet101_Weights

from example_fastapi.resnet_model import ResnetModel
from fast_dynamic_batcher.dyn_batcher import DynBatcher


app = FastAPI()

dyn_batcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    RunVar("_default_thread_limiter").set(CapacityLimiter(16))
    global dyn_batcher
    dyn_batcher = DynBatcher(ResnetModel, 8, 0.1)
    yield
    print("FastAPI shutdown")
    dyn_batcher.stop()


app = FastAPI(lifespan=lifespan)


tform = ResNet101_Weights.IMAGENET1K_V2.transforms()


@app.post("/predict/")
async def predict(
    mode: Annotated[str, Form()],
    size: Annotated[tuple[int, int], Form()],
    file: UploadFile,
):
    dog_image = Image.frombytes(mode, size, await file.read())
    # serializing the PIL image is slow
    # so we preprocess the image here
    formed_image = tform(dog_image).unsqueeze(0)
    return await dyn_batcher.process_batched(formed_image)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
