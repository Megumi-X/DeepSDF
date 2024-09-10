from export_video import export_gif
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--name", "-n", required=True, type=str)
args = parser.parse_args()

folder = "/mnt/data1/xiongxy/eval/" + args.name

export_gif(folder, folder + "/" + args.name + ".gif", 10, "", ".png")