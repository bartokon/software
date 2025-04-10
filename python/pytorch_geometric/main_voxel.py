
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch
from torch_geometric.loader import DataLoader
from dataset_torchstudio_small_voxel import ModelNet40_n3
from model_res_voxel import FullNet
from tqdm.auto import tqdm
from metrics import metrics_mse

import numpy as np
import matplotlib.pyplot as plt
from my_voxel import rotate_and_sample_and_voxelize, draw_voxel_pair

def train():
    model.train()
    metrics_train.reset()
    total_loss = 0
    for i, (input_0, output_0, raw_0) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
        optimizer.zero_grad(True)
        logits = model(
            input_0
        )
        loss = criterion(logits, output_0)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) / train_loader.batch_size
        metrics_train.update(logits, output_0)
    metrics_train.append_mem(metrics_train.compute().cpu())

    return (total_loss / (i+1))

@torch.no_grad()
def test():
    model.eval()
    metrics_test.reset()
    total_loss = 0
    for i, (input_0, output_0, raw_0) in enumerate(tqdm(test_loader, desc="Test", leave=False)):
        logits = model(
            input_0
        )
        loss = criterion(logits, output_0)
        total_loss += float(loss) / test_loader.batch_size
        metrics_test.update(logits, output_0)
    metrics_test.append_mem(metrics_test.compute().cpu())
    return (total_loss / (i+1)), torch.abs(logits-output_0)

def generate_non_repeating_integers_numpy(count):
  # Generate a random permutation of the range
  permutation = np.random.permutation(count)
  return permutation

if __name__=="__main__":
    #torch.autograd.set_detect_anomaly(True)
    train_dataset = ModelNet40_n3(root="data_train")
    test_dataset = ModelNet40_n3(root="data_test")
    if (0):
        samples = 640
        train_dataset = torch.utils.data.Subset(train_dataset, generate_non_repeating_integers_numpy(samples))
        test_dataset = torch.utils.data.Subset(test_dataset, generate_non_repeating_integers_numpy(samples))
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 64)

    model = FullNet()
    #model = torch.load("models/model_23.mod")
    model.to(device='cuda')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, amsgrad=True)
    criterion = torch.nn.MSELoss(reduction = 'mean')
    metrics_train = metrics_mse(threshold = 5)
    metrics_test = metrics_mse(threshold = 5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, cooldown=5, patience=5, mode='min')
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

    loss_train_plot, = ax_loss.plot(loss_train, range(0, len(loss_train)), markersize = 8, label="Train")
    loss_test_plot, = ax_loss.plot(loss_train, range(0, len(loss_train)), markersize = 8, label="Test")
    metrics_train_plot, = ax_metrics.plot(metrics_train.get_mem(), range(0, len(metrics_train.get_mem())), markersize = 8, label="Train")
    metrics_test_plot, = ax_metrics.plot(metrics_test.get_mem(), range(0, len(metrics_test.get_mem())), markersize = 8, label="Test")

    ax_loss.legend()
    ax_metrics.legend()

    for epoch in tqdm(range(1, epochs), desc="Epoch", leave=True):
        loss = train()
        loss_train.append(loss)
        scheduler.step(loss)

        loss, test_acc = test()
        loss_test.append(loss)
        tqdm.write(f'\nEpoch: {epoch:02d}, Loss: {loss_train[-1]:.4f}\nMetrics: {metrics_test.compute():.4f}\nACC: {torch.mean(test_acc, dim = 0)}\n')

        loss_train_plot.set_xdata(range(0, len(loss_train))[1::])
        loss_train_plot.set_ydata(torch.clamp_max(torch.tensor(loss_train), 1e10)[1::])
        loss_test_plot.set_xdata(range(0, len(loss_test))[1::])
        loss_test_plot.set_ydata(torch.clamp_max(torch.tensor(loss_test), 1e10)[1::])
        metrics_train_plot.set_xdata(range(0, len(metrics_train.get_mem()))[1::])
        metrics_train_plot.set_ydata(metrics_train.get_mem()[1::])
        metrics_test_plot.set_xdata(range(0, len(metrics_test.get_mem()))[1::])
        metrics_test_plot.set_ydata(metrics_test.get_mem()[1::])

        ax_loss.relim()
        ax_loss.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        dt = test_dataset[0]
        with torch.no_grad():
            logits = model(torch.unsqueeze(dt[0], dim=0))
        plt.ioff()
        draw_voxel_pair(
            dt[0][1],
            rotate_and_sample_and_voxelize(dt[2], torch.squeeze(logits), 2**18, 2**5),
            epoch
        )
        plt.ion()
        torch.save(model, f"model_checkpoint/model_{epoch}.mod")


    fig.savefig(f"logs/log_{epoch}.pdf")
    plt.waitforbuttonpress()
    torch.save(model, "model_checkpoint/model_last.mod")