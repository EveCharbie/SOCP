import numpy as np
import pickle



# # Run this with env bioviz_and_pyorerun (conda install -c conda-forge bioviz pyorerun rerun-sdk=0.27.2)
# from bioviz import Viz, Kinogram
#
# # b = Viz(
# #     model_path="socp/models/vertebrate_arm_model_with_mesh.bioMod",
# #     show_meshes=True,
# #     mesh_opacity=1.0,
# #     show_global_center_of_mass=False,
# #     show_gravity_vector=False,
# #     show_floor=False,
# #     show_segments_center_of_mass=False,
# #     show_global_ref_frame=True,
# #     show_local_ref_frame=False,
# #     background_color=(1.0, 1.0, 1.0),
# # )
# b = Kinogram(
#     model_path="socp/models/vertebrate_arm_model_with_mesh.bioMod",
#     show_meshes=True,
#     mesh_opacity=1.0,
#     show_global_center_of_mass=False,
#     show_gravity_vector=False,
#     show_floor=False,
#     show_segments_center_of_mass=False,
#     show_global_ref_frame=False,
#     show_local_ref_frame=False,
#     background_color=(1.0, 1.0, 1.0),
# )
#
# with open("/home/charbie/Documents/Programmation/SOCP/results/to_analyze/VertebrateArm_DirectMultipleShooting_NoiseDiscretization_050_CVG_1p0e-08_2026-04-10-12-48_.pkl", "rb") as f:
#     data_DirectMultipleShooting_Noise = pickle.load(f)
#
#
# b.load_movement(
#     np.vstack((
#         np.ones((data_DirectMultipleShooting_Noise["states_opt_mean"].shape[1], )) * np.pi/2,
#         data_DirectMultipleShooting_Noise["states_opt_mean"][:2, :],
#     ))
# )
# # b.set_camera_zoom(0.35)
# b.exec(frame_step=1, figsize=(15, 15), save_path="kinogram/vertebrate_arm.png")





# Run this with env bioviz_and_pyorerun (conda install -c conda-forge bioviz pyorerun rerun-sdk=0.27.2)
from bioviz import Viz, Kinogram

b = Viz(
    model_path="socp/models/somersault_model.bioMod",
    show_meshes=True,
    mesh_opacity=1.0,
    show_global_center_of_mass=False,
    show_gravity_vector=False,
    show_floor=False,
    show_segments_center_of_mass=False,
    show_global_ref_frame=True,
    show_local_ref_frame=False,
    background_color=(1.0, 1.0, 1.0),
)
# b = Kinogram(
#     model_path="socp/models/vertebrate_arm_model_with_mesh.bioMod",
#     show_meshes=True,
#     mesh_opacity=1.0,
#     show_global_center_of_mass=False,
#     show_gravity_vector=False,
#     show_floor=False,
#     show_segments_center_of_mass=False,
#     show_global_ref_frame=False,
#     show_local_ref_frame=False,
#     background_color=(1.0, 1.0, 1.0),
# )

nb_q = 7
with open("/home/charbie/Documents/Programmation/SOCP/results/Somersault_DirectMultipleShooting_NoiseDiscretization_010_DVG_1p0e-06_2026-04-20-05-41_.pkl", "rb") as f:
    data_DirectMultipleShooting_Noise = pickle.load(f)

b.load_movement(np.repeat(data_DirectMultipleShooting_Noise["states_opt_mean"][:nb_q, :], 10, axis=1))

# b.set_camera_zoom(0.35)
# b.exec(frame_step=1, figsize=(15, 15), save_path="kinogram/vertebrate_arm.png")
b.exec()