import torch.nn.functional as F
import torch

class metrics_mse():
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.mem = []
        self.reset()

    def append_mem(self, val):
        self.mem.append(val)

    def get_mem(self):
        return self.mem

    def reset_mem(self):
        self.mem = []

    def update(self, preds, target):
        correct = torch.less(torch.abs(preds-target.float()),self.threshold).view(-1)
        self.num_correct += torch.sum(correct)
        self.num_samples += torch.tensor(correct.shape[0])

    def compute(self):
        return self.num_correct.float() / self.num_samples.float()

    def reset(self):
        self.num_correct = 0
        self.num_samples = 0