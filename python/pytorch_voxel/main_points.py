
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from metrics import metrics
from metrics import visualizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pathlib
from dataset_points import ModelNet40_aligned
from models.transformer import PointCloudTransformer

def train_epoch(loader, metrics, description):
    model.train()
    total_loss = 0
    for i, (input_0, output_0) in enumerate(
        tqdm(loader, desc = description, leave = False)
    ):
        optimizer.zero_grad(True)
        logits = model(
            input_0
        )
        loss = criterion(logits, output_0[1].to(device = 'cuda').double())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() / loader.batch_size
        metrics.update(logits.detach().cpu(), output_0[1].cpu())
    metrics.compute()
    return (total_loss / (i+1))

def eval_epoch(loader, metrics, description):
    model.eval()
    total_loss = 0
    for i, (input_0, output_0) in enumerate(
        tqdm(loader, desc = description, leave = False)
    ):
        logits = model(
            input_0
        )
        loss = criterion(logits, output_0[1].to(device = 'cuda').double())
        total_loss += loss.item() / loader.batch_size
        metrics.update(logits.detach().cpu(), output_0[1].cpu())
        v.update(logits.detach().cpu(), output_0[1].cpu())
    metrics.compute()
    v.visualize(ax_hist)
    v.zero_mem()
    return (total_loss / (i+1))

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
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    model = PointCloudTransformer(embed_dim=128, num_heads=8)
    model.to(device='cuda')

    v = visualizer(train_dataset.classes_lut)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0005, amsgrad = True)
    criterion = torch.nn.CrossEntropyLoss(train_dataset.get_weights())
    metrics_train = metrics(threshold = 0.5)
    metrics_test = metrics(threshold = 0.5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        factor = 0.1,
        cooldown = 0,
        patience = 10,
        mode='min'
    )
    epochs = 101
    loss_train, loss_test = [], []

    plt.ion()
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(2, 2)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[:, 1])

    ax_loss.minorticks_on()
    ax_loss.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax_loss.grid(which='major', linestyle='-', linewidth='1.0', color='black')

    ax_loss.set(
        xlabel="Epoch",
        ylabel="Loss",
        yscale='log',
        xlim = (0, epochs),
        xticks = (range(0, epochs, 5))
    )

    ax_metrics.minorticks_on()
    ax_metrics.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax_metrics.grid(which='major', linestyle='-', linewidth='1.0', color='black')

    ax_metrics.set(
        xlabel="Epoch",
        ylabel="Metrics",
        xlim = (0, epochs),
        ylim = (0, 1),
        xticks = (range(0, epochs, 5))
    )

    loss_train_plot, = ax_loss.plot(
        range(0, len(loss_train)),
        loss_train,
        label="Train",
        markersize = 8,
    )

    loss_test_plot, = ax_loss.plot(
        range(0, len(loss_test)),
        loss_test,
        label="Test",
        markersize = 8,
    )

    metrics_accuracy_train_plot, = ax_metrics.plot(
        range(0, len(metrics_train.mem_accuracy)),
        metrics_train.mem_accuracy,
        label = "Accuracy",
        markersize = 8
    )

    metrics_recall_train_plot, = ax_metrics.plot(
        range(0, len(metrics_train.mem_recall)),
        metrics_train.mem_recall,
        label = "Recall",
        markersize = 8
    )

    metrics_f1Score_train_plot, = ax_metrics.plot(
        range(0, len(metrics_train.mem_f1Score)),
        metrics_train.mem_f1Score,
        label = "F1Score",
        markersize = 8
    )

    ax_loss.legend()
    ax_metrics.legend()

    ax_hist.set(
        xlabel = "Preds",
        ylabel = "Classes",
        xticks = (range(40))
    )

    ax_hist.tick_params(
        top = True, labeltop = True,
        right = True, labelright = True
    )

    ax_hist.set_xticklabels(v.lut.keys(), rotation='vertical')

    for epoch in tqdm(range(1, epochs), desc = "Epoch", leave = True):
        loss_train.append(train_epoch(train_loader, metrics_train, "Train"))
        scheduler.step(loss_train[-1])
        loss_test.append(eval_epoch(test_loader, metrics_test, "Test"))

        tqdm.write(f"\nLR: {scheduler.get_last_lr()}")
        tqdm.write(f"Epoch: {epoch:02d}")
        tqdm.write(f"Loss: {loss_train[-1]:.4f}")
        tqdm.write(f"Metrics: {metrics_test.compute()}")

        loss_train_plot.set_data(
            range(0, len(loss_train)),
            loss_train,
        )
        loss_test_plot.set_data(
            range(0, len(loss_test)),
            loss_test,
        )
        metrics_accuracy_train_plot.set_data(
           range(0, len(metrics_train.mem_accuracy)),
           metrics_train.mem_accuracy,
        )
        metrics_recall_train_plot.set_data(
            range(0, len(metrics_train.mem_recall)),
            metrics_train.mem_recall,
        )
        metrics_f1Score_train_plot.set_data(
            range(0, len(metrics_train.mem_f1Score)),
            metrics_train.mem_f1Score,
        )

        ax_loss.relim(); ax_loss.autoscale_view()
        ax_metrics.relim(); ax_metrics.autoscale_view()
        fig.canvas.draw_idle(); fig.canvas.flush_events()

        os.makedirs("logs", exist_ok = True); fig.savefig(f"logs/log_{epoch}.pdf")
        os.makedirs("model_checkpoint", exist_ok = True)
        torch.save(model, f"model_checkpoint/model_{epoch}.mod")
    plt.waitforbuttonpress()