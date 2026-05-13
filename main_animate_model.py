import numpy as np
import pickle

# Run this with env bioviz_and_pyorerun (conda install -c conda-forge bioviz pyorerun rerun-sdk=0.27.2)
from bioviz import Viz, Kinogram


model_path = "socp/models/squat_model.bioMod"
nb_q = 12

b = Viz(
    model_path=model_path,
    show_meshes=True,
    mesh_opacity=1.0,
    show_global_center_of_mass=True,
    show_gravity_vector=True,
    show_floor=True,
    show_segments_center_of_mass=True,
    show_global_ref_frame=True,
    show_local_ref_frame=True,
    # background_color=(1.0, 1.0, 1.0),
)
b.load_movement(np.zeros((nb_q, 1)))
b.exec()
