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
        correct = []
        for b in range(len(target)):
            for i in range(len(target[b])):
                if (target[b][i] == 1):
                        correct.append(preds[b][i] >= (1 - self.threshold))
        self.num_correct += torch.sum(torch.tensor(correct))
        self.num_samples += torch.tensor(len(correct))

    def compute(self):
        return self.num_correct.float() / self.num_samples.float()

    def reset(self):
        self.num_correct = 0
        self.num_samples = 0