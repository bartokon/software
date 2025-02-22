
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch_geometric.loader import DataLoader
from dataset_torchstudio import ModelNet40_n3
from model_torchstudio import FullNet
from tqdm import tqdm
from metrics import metrics_mse

import matplotlib.pyplot as plt

def train():
    model.train()
    total_loss = 0
    for i, (input_0, input_1, output_0) in enumerate(tqdm(train_loader, desc="Train")):
        optimizer.zero_grad()
        input_0.to(device='cuda')
        input_1.to(device='cuda')
        output_0.to(device='cuda')
        logits = model(
            pos_0 = input_0.pos,
            edge_index_0 = input_0.edge_index,
            batch_0 = input_0.batch,
            pos_1 = input_1.pos,
            edge_index_1 = input_1.edge_index,
            batch_1 = input_1.batch
        )
        loss = criterion(logits, output_0)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    #tqdm.write(f"{torch.abs(torch.sub(logits, output_0))}")
    #print(f"{logits}")
    #print(f"{output_0}")
    #print(f"{torch.sub(logits, output_0)}")
    return (total_loss / train_loader.batch_size)

@torch.no_grad()
def test():
    model.eval()
    for i, (input_0, input_1, output_0) in enumerate(tqdm(test_loader, desc="Test")):
        input_0.to(device='cuda')
        input_1.to(device='cuda')
        output_0.to(device='cuda')
        logits = model(
            pos_0 = input_0.pos,
            edge_index_0 = input_0.edge_index,
            batch_0 = input_0.batch,
            pos_1 = input_1.pos,
            edge_index_1 = input_1.edge_index,
            batch_1 = input_1.batch
        )
        metrics.update(logits, output_0)
    return torch.abs(logits-output_0), metrics.compute()

if __name__=="__main__":
    #torch.autograd.set_detect_anomaly(True)
    train_dataset = ModelNet40_n3(root="data_train")
    test_dataset = ModelNet40_n3(root="data_test")
    evens = list(range(0, len(train_dataset), 256))
    odds  = list(range(0, len(test_dataset),  256))
    train_dataset = torch.utils.data.Subset(train_dataset, evens)
    test_dataset = torch.utils.data.Subset(test_dataset, odds)
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64)

    model = FullNet()
    model.to(device='cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    criterion = torch.nn.MSELoss(reduction = 'mean')
    metrics = metrics_mse(threshold = 5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    epochs = 51

    plt.ion()
    fig = plt.figure()

    ax_loss = fig.add_subplot(211)
    ax_metrics = fig.add_subplot(212)

    ax_loss.grid()
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_xlim(0, epochs)
    ax_loss.set_xticks(range(0, epochs))
    ax_loss.set_ylabel("Loss")
    ax_loss.set_ylim(bottom = 0, top = None, auto = True)

    ax_metrics.grid()
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_xlim(0, epochs)
    ax_metrics.set_xticks(range(0, epochs))
    ax_metrics.set_ylabel("Metrics")
    ax_metrics.set_ylim(0, 1)

    metric_l = [0]
    loss_l = [0]

    loss_plot, = ax_loss.plot(loss_l, range(0, len(loss_l)))
    metrics_plot, = ax_metrics.plot(metric_l, range(0, len(metric_l)))

    #loss_plot, = ax_loss.plot()
    #metrics_plot, = ax_metrics.plot()

    for epoch in tqdm(range(1, epochs), desc="Epoch"):
        loss = train()
        test_acc, met = test()
        #scheduler.step()
        tqdm.write(f'\nEpoch: {epoch:02d}, Loss: {loss:.4f}\nMetrics: {met:.4f}\nACC: {torch.mean(test_acc, dim = 1)}\n')
        loss_l.append(loss)
        metric_l.append(met.cpu())
        metrics.reset()

        loss_plot.set_xdata(range(0, len(loss_l)))
        loss_plot.set_ydata(loss_l)
        metrics_plot.set_xdata(range(0, len(metric_l)))
        metrics_plot.set_ydata(metric_l)

        ax_loss.relim()
        ax_loss.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
    fig.savefig("log.pdf")