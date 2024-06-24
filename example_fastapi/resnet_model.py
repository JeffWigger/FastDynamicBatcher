import time

import torch

from torchvision import models
from torchvision.models import ResNet101_Weights

from fast_dynamic_batcher.dyn_batcher import Task
from fast_dynamic_batcher.inference_template import InferenceModel


class ResnetModel(InferenceModel):
    def __init__(self):
        super().__init__()
        self.tform = ResNet101_Weights.IMAGENET1K_V2.transforms()
        self.smax = torch.nn.Softmax(dim=1)
        self.resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.resnet.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.resnet.to(self.device)

    def infer(self, tasks: list[Task]) -> list[Task]:
        print(len(tasks))
        start_s = time.time()
        inputs = [t.content for t in tasks]
        input_vec = torch.cat(inputs, dim=0)
        with torch.no_grad():
            input_vec = input_vec.to(self.device)
            self.resnet.to(self.device)
            output = self.resnet(input_vec)
        processed = torch.topk(self.smax(output), 1)
        return_values = [
            Task(id=tasks[i].id, content=(processed[0][i].item(), processed[1][i].item())) for i in range(len(tasks))
        ]
        print(f"Execution of single batch took {time.time() - start_s} seconds.")
        return return_values
