
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch_geometric.loader import DataLoader
from model_pn import MLP
from mymodel_0 import PointNetWeird
from dataset import ModelNet40
from tqdm import tqdm

def save(model, epoch, path = "model.pt", loss = 0.01):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },
        path
    )

if __name__=="__main__":
    def train():
        model.train()
        lambda_l1 = 0.001
        total_loss = 0
        for i, data in enumerate(tqdm(train_loader, desc="Train")):
            optimizer.zero_grad()
            logits = model(data.pos)
            loss = criterion(logits, data.y)
            #l1_norim = sum(p.abs().sum() for p in model.parameters())
            #loss += lambda_l1 * l1_norm
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        return (total_loss / len(train_loader.dataset))

    @torch.no_grad()
    def test():
        model.eval()
        total_error = 0

        for i, data in enumerate(tqdm(test_loader, desc="Test")):
            logits = model(data.pos)
            pred = logits
            #print(f"{pred} vs {data.y}")
            #Remake?
            error = torch.abs(torch.abs(pred) - torch.abs(data.y))
            #print(error)
            total_error += error
        return (total_error / len(test_loader.dataset))

    train_dataset = ModelNet40(root="data_train")
    test_dataset = ModelNet40(root="data_test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = PointNetWeird()
    #model = torch.load("models/model.pt")
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in tqdm(range(1, 51), desc="Epoch"):
        loss = train()
        test_acc = test()
        print(f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}, AVG error: [{test_acc}]\n')
        if (epoch % 1 == 0):
            print(f"Model saved. Epoch: {epoch}")
            torch.save(model, "models/model.pt")