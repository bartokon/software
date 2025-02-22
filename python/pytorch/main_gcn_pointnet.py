
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch.utils.data import Dataset, DataLoader
from os.path import isfile
from dataset_pytorch import ModelNet40_n3
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rotate import rotatete
from histogram_model import FullNet

if __name__=="__main__":
    def train():
        model.train()
        total_loss = 0
        for i, (pc_0, pc_1, y) in enumerate(tqdm(train_loader, desc="Train")):
            optimizer.zero_grad()
            logits = model(pc_0, pc_1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        return (total_loss / train_loader.batch_size)

    @torch.no_grad()
    def test():
        model.eval()
        total_error = 0
        for i, (pc_0, pc_1, y) in enumerate(tqdm(test_loader, desc="Test")):
            logits = model(pc_0, pc_1)
            pred = logits
            error = torch.abs(torch.abs(pred) - torch.abs(y))
            total_error += error
        if epoch % 10 == 0:
            figs = []
            for a, b, xyz in zip(pc_0, pc_1, pred):
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.scatter(b[0].cpu(), b[1].cpu(), b[2].cpu(), s = 10, alpha = 0.1)
                r = a
                r = rotatete(r, xyz[0], axis=0)
                r = rotatete(r, xyz[1], axis=1)
                r = rotatete(r, xyz[2], axis=2)
                r = r - torch.mean(r, dim = 1, keepdim=True)
                r = r.cpu()
                ax.scatter(r[0], r[1], r[2], s = 0.5, alpha = 1)
                figs.append(fig)
            with PdfPages(f'pdfs/output_{epoch}.pdf') as pdf:
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight')
            plt.close('all')
        return (total_error / test_loader.batch_size)

    for max_rot in range(0, 45, 10):
        train_dataset = ModelNet40_n3(root="data_train", max_rot = max_rot)
        test_dataset = ModelNet40_n3(root="data_test", max_rot = max_rot)

        train_loader = DataLoader(train_dataset, batch_size = 10, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size = 10)

        if (isfile(f"models/model_gcn_20.pt")):
            model = torch.load(f"models/model_gcn_20.pt")
        else:
            model = FullNet()
        model.to('cuda')

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, amsgrad=True)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        for epoch in tqdm(range(1, 21), desc="Epoch"):
            loss = train()
            test_acc = test()
            #scheduler.step()
            print(f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}, AVG error: \n{test_acc}\n')
            if (epoch % 20 == 0):
                print(f"Model saved. Epoch: {epoch}")
                torch.save(model, f"models/model_gcn_{epoch}.pt")