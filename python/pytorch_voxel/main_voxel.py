
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from metrics import metrics_mse
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from dataset_voxel import ModelNet40_aligned
from models.dense_voxel import FullNet

def train():
    model.train()
    metrics_train.reset()
    total_loss = 0
    for i, (input_0, output_0) in enumerate(
        tqdm(train_loader, desc = "Train", leave = False)
    ):
        optimizer.zero_grad(True)
        logits = model(
            input_0
        )
        to_criterion = output_0[1].to(device = 'cuda').double()
        loss = criterion(logits, to_criterion)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) / train_loader.batch_size
        metrics_train.update(logits, to_criterion)
    metrics_train.append_mem(metrics_train.compute().cpu())

    return (total_loss / (i+1))

@torch.no_grad()
def test():
    model.eval()
    metrics_test.reset()
    total_loss = 0
    for i, (input_0, output_0) in enumerate(
        tqdm(test_loader, desc = "Test", leave=False)
    ):
        logits = model(
            input_0
        )
        to_criterion = output_0[1].to(device = 'cuda').double()
        loss = criterion(logits, to_criterion)
        total_loss += float(loss) / test_loader.batch_size
        metrics_test.update(logits, to_criterion)
    metrics_test.append_mem(metrics_test.compute().cpu())
    return (total_loss / (i + 1)), torch.abs(logits - to_criterion)

if __name__=="__main__":
    torch.autograd.set_detect_anomaly(True)
    classes = [f.name for f in pathlib.Path("dataset").iterdir() if f.is_dir()]

    train_dataset = ModelNet40_aligned(
        root = "dataset",
        classes = classes,
        train = True,
        transform = None,
        target_transform = None,
        download = True
    )
    test_dataset = ModelNet40_aligned(
        root = "dataset",
        classes = classes,
        train = False,
        transform = None,
        target_transform = None,
        download = True
    )
    weight = train_dataset.get_weights()
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 128)

    model = FullNet()
    model.to(device='cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.1, amsgrad = True)
    criterion = torch.nn.CrossEntropyLoss(weight = weight)
    metrics_train = metrics_mse(threshold = 0.1)
    metrics_test = metrics_mse(threshold = 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        factor = 0.5,
        cooldown = 5,
        patience = 5,
        mode='min'
    )
    epochs = 101

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
    ax_loss.set_yscale('log')

    ax_metrics.grid()
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_xlim(0, epochs)
    ax_metrics.set_xticks(range(0, epochs))
    ax_metrics.set_ylabel("Metrics")
    ax_metrics.set_ylim(0, 1)
    loss_train = []
    loss_test = []

    loss_train_plot, = ax_loss.plot(
        loss_train,
        range(0, len(loss_train)),
        markersize = 8,
        label="Train"
    )
    loss_test_plot, = ax_loss.plot(
        loss_train,
        range(0, len(loss_train)),
        markersize = 8,
        label="Test"
    )
    metrics_train_plot, = ax_metrics.plot(
        metrics_train.get_mem(),
        range(0, len(metrics_train.get_mem())),
        markersize = 8,
        label="Train"
    )
    metrics_test_plot, = ax_metrics.plot(
        metrics_test.get_mem(),
        range(0, len(metrics_test.get_mem())),
        markersize = 8,
        label="Test"
    )

    ax_loss.legend()
    ax_metrics.legend()

    for epoch in tqdm(range(1, epochs), desc = "Epoch", leave = True):
        loss = train()
        loss_train.append(loss)
        scheduler.step(loss)
        loss, test_acc = test()
        loss_test.append(loss)
        tqdm.write(f"\nLR: {scheduler.get_last_lr()}")
        tqdm.write(f"Epoch: {epoch:02d}")
        tqdm.write(f"Loss: {loss_train[-1]:.4f}")
        tqdm.write(f"Metrics: {metrics_test.compute():.4f}")
        tqdm.write(
            f"ACC: {torch.round(torch.mean(test_acc, dim = 0), decimals = 2)}"
        )

        loss_train_plot.set_xdata(range(0, len(loss_train))[1::])
        loss_train_plot.set_ydata(\
            torch.clamp_max(torch.tensor(loss_train), 1e10)[1::]
        )
        loss_test_plot.set_xdata(range(0, len(loss_test))[1::])
        loss_test_plot.set_ydata(\
            torch.clamp_max(torch.tensor(loss_test), 1e10)[1::]
        )
        metrics_train_plot.set_xdata(\
            range(0, len(metrics_train.get_mem()))[1::]
        )
        metrics_train_plot.set_ydata(metrics_train.get_mem()[1::])
        metrics_test_plot.set_xdata(
            range(0, len(metrics_test.get_mem()))[1::]\
        )
        metrics_test_plot.set_ydata(metrics_test.get_mem()[1::])

        ax_loss.relim()
        ax_loss.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.ion()

    os.makedirs("logs", exist_ok = True)
    fig.savefig(f"logs/log_{epoch}.pdf")
    plt.waitforbuttonpress()
    os.makedirs("model_checkpoint", exist_ok = True)
    torch.save(model, "model_checkpoint/model_last.mod")