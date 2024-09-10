import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torch.optim as optim
import open3d as o3d
from pbrt_renderer import PbrtRenderer
from config import to_real_array
from log import create_folder
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network_size = 1024

class DeepSDFWithCode(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(DeepSDFWithCode, self).__init__()
        self.network = nn.Sequential(
            weight_norm(nn.Linear(3, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, network_size)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            weight_norm(nn.Linear(network_size, 1)),
        )
    
    def forward(self, coords):
        return self.network(coords)


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", "-n", required=True, type=str)
    parser.add_argument("--xc", "-x", default=-0.1, type=float)
    parser.add_argument("--zc", "-z", default=0.1, type=float)
    args = parser.parse_args()
    name = args.name

    pcd_out = o3d.io.read_point_cloud("/mnt/data1/xiongxy/pcd/" + name + "/point_cloud_downsampled.ply")
    pcd_in = o3d.io.read_point_cloud("/mnt/data1/xiongxy/pcd/" + name + "/" + name + "_inner.ply")
    vertices = np.vstack([np.asarray(pcd_out.points), np.asarray(pcd_in.points)])
    vertices -= np.average(vertices, axis=0)
    vertices -= np.max(vertices, axis=0) * 1.5 * np.array([0, 1, 0])
    out_num = np.asarray(pcd_out.points).shape[0]
    in_num = np.asarray(pcd_in.points).shape[0]
    v = torch.tensor(vertices, dtype=torch.float32, device=device)
    rho = torch.cat([torch.ones(out_num, 1), torch.ones(in_num, 1) * 40], dim=0).to(device)

    # x = np.load("/mnt/data1/xiongxy/pcd/" + name + "/result.npy")
    # x = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
    # target_in_out = torch.cat([torch.ones(out_num, 1).to(device), torch.tanh(x * 3) * 0.5 + 0.5], dim=0).to(device)

    num_epochs = 10000
    model_folder = "/mnt/data1/xiongxy/model/" + name
    create_folder(model_folder, exist_ok=True)

    model = DeepSDFWithCode().to(device)
    torch.save(model.state_dict(), model_folder + "/model_0.pth")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    losses = []
    losses_100 = []
    def closure(epoch):
        optimizer.zero_grad()
        loss = 0
        outputs = model(v)
        # loss += torch.norm(outputs - target_sdf) ** 2
        in_out = torch.tanh(outputs * 5) / 2 + 0.5
        center = torch.sum(v * in_out * rho, axis=0) / torch.clamp(torch.sum(in_out * rho, axis=0), 1e-8)
        loss += ((center[0] - args.xc) ** 2 + (center[2] - args.zc) ** 2) * 20
        # loss += torch.norm(in_out - target_in_out) ** 2
        loss += torch.norm(nn.functional.relu(-in_out[:out_num] + .7)) ** 2 * 3e-3
        loss.backward()
        
        losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        if (epoch + 1) % 100 == 0:
            losses_100.append(loss.item())
            torch.save(model.state_dict(), model_folder + f"/model_{epoch+1}.pth")
        return loss

    for epoch in range(num_epochs):
        optimizer.step(closure=lambda: closure(epoch))

    min_loss = min(losses_100)
    min_loss_index = (losses_100.index(min_loss) + 1) * 100
    np.save(model_folder + "/min_loss_index.npy", min_loss_index)

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(model_folder + "/loss.png")

if __name__ == "__main__":
    main()