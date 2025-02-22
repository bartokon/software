import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch_geometric
import torch
import pickle
from torch_geometric.loader import DataLoader
from mymodel_1 import PointNet_GCN_rotation
from dataset import ModelNet40graph
from tqdm import tqdm
from torch_geometric.transforms import Compose
from rotate import rotate

if __name__=="__main__":
    def train():
        model.train()
        total_loss = 0
        for i, data in enumerate(tqdm(train_loader, desc="Train")):
            optimizer.zero_grad()
            logits = model(
                data[0].pos, data[0].edge_index,
                data[1].pos, data[1].edge_index,
            )
            loss = criterion(logits, data[0].pos)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        return (total_loss / len(train_loader.dataset))

    @torch.no_grad()
    def test():
        model.eval()
        total_error = 0

        for i, data in enumerate(tqdm(test_loader, desc="Test")):
            logits = model(
                data[0].pos, data[0].edge_index,
                data[1].pos, data[1].edge_index,
            )
            pred = logits
            error = torch.abs(torch.abs(pred) - torch.abs(data[0].pos))
            total_error += error
        return (total_error / len(test_loader.dataset))

    train_dataset = ModelNet40graph(root="data_train")
    test_dataset = ModelNet40graph(root="data_test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = PointNet_GCN_rotation()
    model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in tqdm(range(1, 1001), desc="Epoch"):
        loss = train()
        test_acc = test()
        #scheduler.step()
        print(f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}, AVG error: [{test_acc}]\n')
        #if (epoch % 1 == 0):
            #print(f"Model saved. Epoch: {epoch}")
            #save(model, "models/model_gcn.pt")