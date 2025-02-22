
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch_geometric
import torch
import pickle
from torch_geometric.loader import DataLoader
from mymodel_1 import PointNet_GCN
from dataset import ModelNet40graph, ModelNet40graph_i
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rotate import rotate

if __name__=="__main__":
    def train():
        model.train()
        total_loss = 0
        batch = 1
        optimizer.zero_grad()
        for i, data in enumerate(tqdm(train_loader, desc="Train")):
            logits = model(
                data[0].pos, data[0].edge_index,
                data[1].pos, data[1].edge_index,
            )
            loss = criterion(logits, data[1].y)
            loss.backward()
            if (i % batch == 0):
                optimizer.step()
                optimizer.zero_grad()
            total_loss += float(loss)
        return (total_loss / len(train_loader.dataset))

    @torch.no_grad()
    def test():
        model.eval()
        total_error = 0
        figs = []
        for i, data in enumerate(tqdm(test_loader, desc="Test")):
            #myplot(data[0])
            logits = model(
                data[0].pos, data[0].edge_index,
                data[1].pos, data[1].edge_index,
            )
            pred = logits
            error = torch.abs(torch.abs(pred) - torch.abs(data[1].y))
            total_error += error
            if epoch > 10:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.scatter(data[0].pos[:, 0].cpu(), data[0].pos[:, 1].cpu(), data[0].pos[:, 2].cpu())
                #xyz = data[0].y.cpu()
                xyz = pred
                r = data[0].pos
                r = rotate(r, xyz[0], axis=0)
                r = rotate(r, xyz[1], axis=1)
                r = rotate(r, xyz[2], axis=2)
                r = r.cpu()
                ax.scatter(r[:, 0], r[:, 1], r[:, 2])
                figs.append(fig)

        if epoch > 10:
            with PdfPages(f'output_{epoch}.pdf') as pdf:
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
            plt.close('all')
        return (total_error / len(test_loader.dataset))


    train_dataset = ModelNet40graph(root="data_train")
    test_dataset = ModelNet40graph(root="data_test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = PointNet_GCN()
    model.to('cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, amsgrad=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in tqdm(range(1, 21), desc="Epoch"):
        loss = train()
        test_acc = test()
        scheduler.step()
        print(f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}, AVG error: [{test_acc}]\n')
        #if (epoch % 1 == 0):
            #print(f"Model saved. Epoch: {epoch}")
            #save(model, "models/model_gcn.pt")