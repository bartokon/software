
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch.utils.data import Dataset, DataLoader

from dataset_pytorch import ModelNet40_n3
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rotate import rotatete
from mymodel_2 import ConvNet

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive

if __name__=="__main__":
    def train():
        model.train()
        total_loss = 0
        for i, (pc_0, pc_1, y) in enumerate(tqdm(train_loader, desc="Train")):
            optimizer.zero_grad()
            logits_0 = model(pc_0)
            logits_1 = model(pc_1)
            loss = criterion(logits_0, logits_1)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        return (total_loss / train_loader.batch_size)

    @torch.no_grad()
    def test():
        model.eval()
        total_error = 0
        for i, (pc_0, pc_1, y) in enumerate(tqdm(test_loader, desc="Test")):
            logits_0 = model(pc_0)
            logits_1 = model(pc_1)
            error = torch.abs(torch.abs(logits_0) - torch.abs(logits_1))
            total_error += error
        return (total_error / test_loader.batch_size)

    train_dataset = ModelNet40_n3(root="data_train")
    test_dataset = ModelNet40_n3(root="data_test")

    train_loader = DataLoader(train_dataset, batch_size = 10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = 10)

    model = ConvNet()
    model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in tqdm(range(1, 51), desc="Epoch"):
        loss = train()
        test_acc = test()
        scheduler.step()
        print(f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}, AVG error: \n{test_acc}\n')
        #print(f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}')
        #if (epoch % 1 == 0):
            #print(f"Model saved. Epoch: {epoch}")
            #save(model, "models/model_gcn.pt")