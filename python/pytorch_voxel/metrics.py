from torcheval.metrics.classification import BinaryRecall
from torcheval.metrics.classification import BinaryAccuracy
from torcheval.metrics.classification import BinaryF1Score

import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class metrics():
    def __init__(self, threshold: float = 0.5):
        self.accuracy = BinaryAccuracy(threshold = threshold)
        self.recall = BinaryRecall(threshold = threshold)
        self.f1Score = BinaryF1Score(threshold = threshold)
        self.mem_accuracy = []
        self.mem_recall = []
        self.mem_f1Score = []

    def update(self, preds, target):
        for b in range(len(target)):
            self.accuracy.update(preds[b], target[b])
            self.recall.update(preds[b], target[b])
            self.f1Score.update(preds[b], target[b])

    def compute(self):
        accuracy = self.accuracy.compute()
        recall = self.recall.compute()
        f1Score = self.f1Score.compute()
        self.mem_accuracy.append(accuracy)
        self.mem_recall.append(recall)
        self.mem_f1Score.append(f1Score)
        return accuracy, recall, f1Score

    def get_mem(self):
        return [self.mem_accuracy, self.mem_recall, self.mem_f1Score]

    def get_len(self):
        return [
            range(0, len(self.mem_accuracy)),
            range(0, len(self.mem_recall)),
            range(0, len(self.mem_f1Score))
        ]

    def reset(self):
        pass

class visualizer():
    def __init__(self, classes_lut):
        self.lut = {}
        for k, (name, bools) in classes_lut.items():
            index = next((i for i, x in enumerate(bools) if x == True), None)
            self.lut[index] = name
        self.zero_mem()
        self.df_mem = []

    def zero_mem(self):
        self.mem = {
            k : {
                    kk : 0
                    for kk in self.lut.values()
                }
            for k in self.lut.values()
        }

    def update(self, preds, target):
        indexes_pred = torch.argmax(preds, dim = 1)
        indexes_target = torch.argmax(target.int(), dim = 1)
        for it, ip in zip(indexes_target, indexes_pred):
            target_class = self.lut[int(it)]
            pred_class = self.lut[int(ip)]
            loaded = self.mem[target_class]
            loaded[pred_class] += 1
            self.mem[target_class] = loaded

    def __str__(self):
        s = ""
        for c in self.mem.items():
            s += f"{c}\n"
        return s

    def visualize(self, ax):
        df = pd.DataFrame.from_dict(self.mem, orient = 'index')
        df = df.apply(lambda row: ((np.exp(row) / np.sum(np.exp(row)))*100), axis = 1).astype(int)
        self.df_mem.append(df)
        ax.clear()
        sns.heatmap(df, annot = True, fmt = 'd', cmap = 'viridis', ax = ax, cbar = False)
        ax.set(
            xlabel = "Preds",
            ylabel = "Classes",
            xticks = (range(len(self.lut.values())))
        )
        ax.tick_params(
            top = True, labeltop = True,
            right = True, labelright = True
        )
        ax.set_xticklabels(self.lut.values(), rotation='vertical')


