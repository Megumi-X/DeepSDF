import torch
import numpy as np
import open3d as o3d
from opt import DeepSDFWithCode
from config import to_real_array
from pbrt_renderer import PbrtRenderer
from log import create_folder
import os
from argparse import ArgumentParser
from export_video import export_gif

parser = ArgumentParser()
parser.add_argument("--name", "-n", required=True, type=str)
parser.add_argument("--y_pos", "-y", default=0, type=float)
parser.add_argument("--axis", "-a", default=2, type=int)
parser.add_argument("--slice_num", "--sn", default=40, type=int)
args = parser.parse_args()
name = args.name
axis = args.axis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pcd_out = o3d.io.read_point_cloud("/mnt/data1/xiongxy/pcd/" + name + "/point_cloud_downsampled.ply")
pcd_in = o3d.io.read_point_cloud("/mnt/data1/xiongxy/pcd/" + name + "/" + name + "_inner.ply")
vertices = np.vstack([np.asarray(pcd_out.points), np.asarray(pcd_in.points)])
vertices -= np.average(vertices, axis=0)
vertices -= np.max(vertices, axis=0) * 1.5 * np.array([0, 1, 0])
v = torch.tensor(vertices, dtype=torch.float32, device=device)
out_num = np.asarray(pcd_out.points).shape[0]

base = torch.ones(v.shape[0], 1).to(device)

model = DeepSDFWithCode().to(device)
try:
    min_loss_index = np.load("/mnt/data1/xiongxy/model/" + name + "/min_loss_index.npy")
except:
    min_loss_index = 10000
model.load_state_dict(torch.load("/mnt/data1/xiongxy/model/" + name + "/model_" + str(min_loss_index) +".pth"))
# model.load_state_dict(torch.load("/mnt/data1/xiongxy/model/" + name + "/model_10000.pth"))


outputs = model(v)
in_out = torch.tanh(outputs * 5) / 2 + 0.5
center = torch.sum(v * in_out, axis=0) / torch.clamp(torch.sum(in_out, axis=0), 1e-5)
print(center)

def visualize(slice_pos, image_name):
    # x = np.linspace(-3, 3, 60)
    # y = np.linspace(-3, 3, 60)
    # z = np.linspace(-3, 3, 60)
    # xx, yy, zz = np.meshgrid(x, y, z)
    # points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
    # angle_input = torch.cat([torch.ones(points.shape[1], 1).to(device) * angle_1, torch.ones(points.shape[1], 1).to(device) * angle_2], dim=1)
    # sdf_values = model(torch.tensor(points.T, dtype=torch.float32, device=device), angle_input)
    # verts, faces, _, _ = measure.marching_cubes(sdf_values.detach().cpu().numpy().reshape((60, 60, 60)), level=0)
    # print("verts: ", verts)
    # print("faces: ", faces)
    outputs = model(v)
    in_outs = torch.tanh(outputs * 50) / 2 + 0.5
    # in_outs[:out_num] = 1

    r = PbrtRenderer()
    if axis == 0:
        eye = to_real_array([args.y_pos - 3, 0.75, -3])
        look_at = to_real_array([args.y_pos, -0.75, 0])
        up = [0, 1, 0]
    else:
        eye = to_real_array([args.y_pos, 0.75, -3])
        look_at = to_real_array([args.y_pos, -0.75, 0])
        up = [0, 0, 1]
    eye = look_at + 1 * (eye - look_at)
    r.set_camera(eye=eye, look_at=look_at, up=up, fov=40)
    r.add_infinite_light({
        "rgb L": (1., 1., 1.)
    })
    for i in range(vertices.shape[0]):
        in_out = in_outs[i].item()
        if vertices[i, 1] < slice_pos:
            r.add_sphere(to_real_array(vertices[i]), 0.007, ("diffuse", { "rgb reflectance": (in_out, 0, 1 - in_out) }))
    # r.add_sphere(center.detach().numpy(), 0.03, ("diffuse", { "rgb reflectance": (1.0, 0.0, 0.0) }))
    # r.add_cylinder(to_real_array([0, 0, 0]), to_real_array([0, 10, 0]), 1e-2, ("diffuse", { "rgb reflectance": (0.0, 0.0, 1.0) }))
    # r.add_triangle_mesh(verts, faces, texture_coords=None, texture_image=None, material=("diffuse", { "rgb reflectance": (0.1, 0.8, 0.1) }))
    # r.add_plane(to_real_array([0., 0., 0.]), to_real_array([0., 1., 0.]), 100., ("diffuse", { "rgb reflectance": (0.5, 0.5, 0.5) }))
    r.set_image(pixel_samples=32, file_name=image_name,
        resolution=[1000, 1000])
    r.render(use_gpu="PBRT_OPTIX7_PATH" in os.environ)

eval_folder = "/mnt/data1/xiongxy/eval/" + name
create_folder(eval_folder, exist_ok=True)

min_pos = np.min(vertices[:, 1])
max_pos = np.max(vertices[:, 1])

for i in range(args.slice_num):
    slice_pos = min_pos + (max_pos - min_pos) * i / (args.slice_num - 1)
    visualize(slice_pos, eval_folder + "/slice_{:02d}.png".format(i))

export_gif(eval_folder, eval_folder + "/" + name + ".gif", 10, "slice_", ".png")