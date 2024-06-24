import asyncio
import concurrent
import time

from urllib.request import urlretrieve

import torch

from PIL import Image
from torchvision import models
from torchvision.models import ResNet101_Weights

from example_fastapi.resnet_model import ResnetModel
from fast_dynamic_batcher.dyn_batcher import DynBatcher


# setup for run_resnet_single
urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
dog_image = Image.open("dog.jpg")
tform = ResNet101_Weights.IMAGENET1K_V2.transforms()
smax = torch.nn.Softmax(dim=1)
resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
resnet.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
resnet.to(device)
formed_image = tform(dog_image)
formed_image = formed_image.unsqueeze(0)


def run_resnet_single(single):
    with torch.no_grad():
        single = single.to(device)
        # print(single.device)
        resnet.to(device)
        output = resnet(single)
    processed = torch.topk(smax(output), 1)
    return (processed[0][0].item(), processed[1][0].item())


dyn_batcher = None


async def batched(n_exp=100):
    loop = asyncio.get_running_loop()
    # By default it uses #num_cores + 4 cores
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=16 * 3))
    start_s = time.time()
    print(f"start batched: {start_s}")
    print(n_exp)
    requests = []
    for _ in range(n_exp):
        requests.append(dyn_batcher.process_batched(formed_image))
    res = await asyncio.gather(*requests)
    print(f"Execution batched took {time.time() - start_s} seconds.")
    print(f"end batched: {time.time()}")
    print(res)


def single(n_exp=100):
    requests = []
    for _ in range(n_exp):
        requests.append(run_resnet_single(formed_image))
    # print(requests)


if __name__ == "__main__":
    dyn_batcher = DynBatcher(ResnetModel, 16, 0.1)
    time.sleep(10)
    start_s = time.time()
    asyncio.run(batched(400))
    print(f"Execution batched took {time.time() - start_s} seconds.")

    start_s = time.time()
    single(400)
    print(f"Execution single took {time.time() - start_s} seconds.")
    dyn_batcher.stop()
    print("stopped")
