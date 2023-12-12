# encoding: utf-8
"""This file includes necessary params, info."""
import os
import os.path as osp
import mmcv
import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(
    root_dir, "output"
)  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "../data")
bop_root = osp.join(data_root, "ROBI/bop")
# ---------------------------------------------------------------- #
# TLESS DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "robi_20000")
train_render_dir = osp.join(dataset_root, "train")
test_dir = osp.join(dataset_root, "test")

# model_dir = osp.join(dataset_root, "models_reconst")  # use recon models as default
model_dir = osp.join(dataset_root, "models")
model_cad = osp.join(dataset_root, "models")
model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 0.001
# object info
objects = [str(i) for i in range(1, 7)]
id2obj = {i: str(i) for i in range(1, 7)}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [
    ((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)
]  # for renderer

# Camera info
tr_render_width = 1280
tr_render_height = 1024
width = te_width = 720  # pbr size
height = te_height = 540  # pbr size
zNear = 0.25
zFar = 6.0
tr_render_center = (tr_render_height / 2, tr_render_width / 2)
te_center = (te_width / 2.0, te_height / 2.0)
zNear = 0.25
zFar = 6.0

# NOTE: for tless, the camera matrix is not fixed!
camera_matrix = np.array(
    [1083.097046, 0, 379.326874, 0, 1083.097046, 509.437195, 0, 0, 1]
).reshape(3, 3)


diameters = np.array([37.321, 29.538, 77.839, 48.432, 32.873, 33.004, 24.860]) / 1000.0


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


# ref core/gdrn_modeling/tools/tless/tless_1_compute_fps.py
def get_fps_points():
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


# ref core/gdrn_modeling/tools/tless/tless_1_compute_keypoints_3d.py
def get_keypoints_3d():
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict
