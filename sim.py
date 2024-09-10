import torch
import numpy as np
import open3d as o3d
from opt import DeepSDFWithCode
from config import to_real_array
from pbrt_renderer import PbrtRenderer
from log import create_folder
import trimesh
from pathlib import Path
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--name", "-n", required=True, type=str)
parser.add_argument("--distance", "-d", required=True, type=float)
parser.add_argument("--texture", "-t", required=False, default="", type=str)
parser.add_argument("--rot_x", "-x", required=False, default=0, type=float)
parser.add_argument("--rot_y", "-y", required=False, default=0, type=float)
parser.add_argument("--rot_z", "-z", required=False, default=0, type=float)
parser.add_argument("--camera_axis", "-ca", required=False, default=0, type=int)
parser.add_argument("--use_mesh_uv", "-u", required=False, default=False, action="store_true")
parser.add_argument("--scene", "-s", required=False, default=False, action="store_true")
parser.add_argument("--final_y_angle", "-fy", required=False, default=0, type=float)
args = parser.parse_args()
name = args.name

root = "/home/xiongxy/GraDy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_final = DeepSDFWithCode().to(device)
try:
    min_loss_index = np.load("/mnt/data1/xiongxy/model/" + name + "/min_loss_index.npy")
except:
    min_loss_index = 10000
model_final.load_state_dict(torch.load("/mnt/data1/xiongxy/model/" + name + "/model_" + str(min_loss_index) +".pth"))
model_init = DeepSDFWithCode().to(device)
model_init.load_state_dict(torch.load("/mnt/data1/xiongxy/model/" + name + "/model_0.pth"))

pcd_out = o3d.io.read_point_cloud("/mnt/data1/xiongxy/pcd/" + name + "/point_cloud_downsampled.ply")
pcd_in = o3d.io.read_point_cloud("/mnt/data1/xiongxy/pcd/" + name + "/" + name + "_inner.ply")
vertices = np.vstack([np.asarray(pcd_out.points), np.asarray(pcd_in.points)])
vertices -= np.average(vertices, axis=0)
vertices -= np.max(vertices, axis=0) * 1.5 * np.array([0, 1, 0])
v = torch.tensor(vertices, dtype=torch.float32, device=device)
out_num = np.asarray(pcd_out.points).shape[0]
in_num = np.asarray(pcd_in.points).shape[0]
rho = torch.cat([torch.ones(out_num, 1), torch.ones(in_num, 1) * 40], dim=0).to(device)

outputs_final = model_final(v)
in_out_final = torch.tanh(outputs_final * 5) / 2 + 0.5
center_final = torch.sum(v * in_out_final * rho, axis=0) / torch.clamp(torch.sum(in_out_final * rho, axis=0), 1e-5)

outputs_init = model_init(v)
in_out_init = torch.tanh(outputs_init * 5) / 2 + 0.5
center_init = torch.sum(v * in_out_init * rho, axis=0) / torch.clamp(torch.sum(in_out_init * rho, axis=0), 1e-5)

distances = torch.sqrt((v[:, 0] - center_final[0])**2 + (v[:, 2] - center_final[2])**2)
cylinder_points = v[distances <= 0.01]
print(cylinder_points.shape)
print(center_final)
print(center_init)
hanging_y = torch.max(cylinder_points, axis=0).values[1]
print(hanging_y)
rot_y_angle = torch.atan2(center_final[2] - center_init[2], center_final[0] - center_init[0]).cpu().detach().numpy() + np.pi
rot_z_angle = -torch.atan2(torch.sqrt((center_final[0] - center_init[0])**2 + (center_final[2] - center_init[2])**2), hanging_y - center_final[1]).cpu().detach().numpy()

x_min = torch.min(v[:, 0])
x_max = torch.max(v[:, 0])
z_min = torch.min(v[:, 2])
z_max = torch.max(v[:, 2])
y_min = torch.min(v[:, 1])
y_max = torch.max(v[:, 1])
x_c_norm = ((center_final[0] - x_min) / (x_max - x_min)).cpu().detach().numpy()
z_c_norm = ((center_final[2] - z_min) / (z_max - z_min)).cpu().detach().numpy()
hanging_y_norm = ((hanging_y - y_min) / (y_max - y_min)).cpu().detach().numpy()

if args.scene:
    mesh = trimesh.load(Path(root) / "asset" / "mesh" / (args.name + ".obj")).dump(concatenate=True)
    mesh_init = trimesh.load(Path(root) / "asset" / "mesh" / (args.name + ".obj")).dump(concatenate=True)
else:
    mesh = trimesh.load(Path(root) / "asset" / "mesh" / (args.name + ".obj"))
    mesh_init = trimesh.load(Path(root) / "asset" / "mesh" / (args.name + ".obj"))
mesh.apply_translation(-mesh.centroid)
mesh_init.apply_translation(-mesh_init.centroid)
mesh_init.apply_transform(trimesh.transformations.rotation_matrix(args.rot_y * np.pi / 180, [0, 1, 0]))
mesh_init.apply_transform(trimesh.transformations.rotation_matrix(args.rot_x * np.pi / 180, [1, 0, 0]))
mesh_init.apply_transform(trimesh.transformations.rotation_matrix(args.rot_z * np.pi / 180, [0, 0, 1]))

r = PbrtRenderer()
if args.camera_axis == 0:
    eye = to_real_array([-2, .3, 0])
elif args.camera_axis == 2:
    eye = to_real_array([0, .3, 2])
else:
    eye = to_real_array([1.414, .3, 1.414])
look_at = to_real_array([0, 0, 0])
eye = look_at + args.distance * (eye - look_at)
r.set_camera(eye=eye, look_at=look_at, up=[0, 1, 0], fov=40)
r.add_infinite_light({
    "rgb L": (1., 1., 1.)
})
mesh_vertices = mesh.vertices
faces = mesh.faces

uv = np.zeros((mesh_vertices.shape[0], 2))
if args.use_mesh_uv:
    uv = mesh.visual.uv
else:
    uv[:, 0] = (mesh_vertices[:, 0] / (np.max(mesh_vertices[:, 0]) - np.min(mesh_vertices[:, 0])) + 0.5) % 1
    uv[:, 1] = (mesh_vertices[:, 1] / (np.max(mesh_vertices[:, 1]) - np.min(mesh_vertices[:, 1])) + 0.5) % 1

sim_folder = "/mnt/data1/xiongxy/sim/" + name
create_folder(sim_folder, exist_ok=True)

mesh.apply_transform(trimesh.transformations.rotation_matrix(args.rot_y * np.pi / 180, [0, 1, 0]))
mesh.apply_transform(trimesh.transformations.rotation_matrix(args.rot_x * np.pi / 180, [1, 0, 0]))
mesh.apply_transform(trimesh.transformations.rotation_matrix(args.rot_z * np.pi / 180, [0, 0, 1]))
mesh_vertices = mesh.vertices

x_min_mesh = np.min(mesh_vertices[:, 0])
x_max_mesh = np.max(mesh_vertices[:, 0])
z_min_mesh = np.min(mesh_vertices[:, 2])
z_max_mesh = np.max(mesh_vertices[:, 2])
y_min_mesh = np.min(mesh_vertices[:, 1])
y_max_mesh = np.max(mesh_vertices[:, 1])

x_c_mesh = (x_max_mesh - x_min_mesh) * x_c_norm + x_min_mesh
z_c_mesh = (z_max_mesh - z_min_mesh) * z_c_norm + z_min_mesh
distances = np.sqrt((mesh_vertices[:, 0] - x_c_mesh)**2 + (mesh_vertices[:, 2] - z_c_mesh)**2)
# hanging_y_mesh = np.max(mesh_vertices[distances <= 0.05 * args.distance, 1])
hanging_y_mesh = (y_max_mesh - y_min_mesh) * hanging_y_norm + y_min_mesh

mesh_init.apply_translation(to_real_array([-x_c_mesh, -hanging_y_mesh, -z_c_mesh]))
mesh_init.apply_transform(trimesh.transformations.rotation_matrix(rot_y_angle, [0, 1, 0]))
mesh_init.apply_transform(trimesh.transformations.rotation_matrix(rot_z_angle, [0, 0, 1]))
mesh_init.apply_transform(trimesh.transformations.rotation_matrix(args.final_y_angle * np.pi / 180, [0, 1, 0]))
mesh_init.apply_translation(to_real_array([x_c_mesh, hanging_y_mesh, z_c_mesh]))

mesh_init_vertices = mesh_init.vertices

if args.texture == "":
    r.add_triangle_mesh(vertices = mesh_vertices, elements=faces, texture_coords=None, texture_image=Path(root) / "asset" / "texture" / "chessboard.jpg", material=("conductor", {"spectrum eta": "metal-Cu-eta", "spectrum k": "metal-Cu-k", "float roughness": 0.1}))
else:
    r.add_triangle_mesh(vertices = mesh_vertices, elements=faces, texture_coords=uv, texture_image=Path(root) / "asset" / "texture" / args.texture, material=("diffuse", {"rgb reflectance": (1, 1, 1)}))
r.add_cylinder(to_real_array([x_c_mesh, 0, z_c_mesh]), to_real_array([x_c_mesh, 40 * args.distance, z_c_mesh]), 0.005 * args.distance, ("conductor", {"spectrum eta": "metal-Au-eta", "spectrum k": "metal-Au-k", "float roughness": 0.001}))
r.add_sphere(to_real_array([x_c_mesh, hanging_y_mesh, z_c_mesh]), 0.01, ("diffuse", { "rgb reflectance": (1.0, 0.0, 0.0) }))
r.set_image(pixel_samples=256, file_name=sim_folder + "/final.png", resolution=[1000, 1000])
r.render(use_gpu="PBRT_OPTIX7_PATH" in os.environ)

r.clear_shapes()
if args.texture == "":
    r.add_triangle_mesh(vertices = mesh_init_vertices, elements=faces, texture_coords=None, texture_image=Path(root) / "asset" / "texture" / "chessboard.jpg", material=("conductor", {"spectrum eta": "metal-Cu-eta", "spectrum k": "metal-Cu-k", "float roughness": 0.1}))
else:
    r.add_triangle_mesh(vertices = mesh_init_vertices, elements=faces, texture_coords=uv, texture_image=Path(root) / "asset" / "texture" / args.texture, material=("diffuse", {"rgb reflectance": (1, 1, 1)}))
r.add_cylinder(to_real_array([x_c_mesh, 0, z_c_mesh]), to_real_array([x_c_mesh, 40 * args.distance, z_c_mesh]), 0.005 * args.distance, ("conductor", {"spectrum eta": "metal-Au-eta", "spectrum k": "metal-Au-k", "float roughness": 0.001}))
r.set_image(pixel_samples=256, file_name=sim_folder + "/init.png", resolution=[1000, 1000])
r.render(use_gpu="PBRT_OPTIX7_PATH" in os.environ)