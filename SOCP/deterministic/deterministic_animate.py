import numpy as np

import pyorerun

from ..utils import ExampleType


def animate_ocp(
        example_type,
        final_time,
        n_shooting,
        q_sol,
        excitations_sol,
):
    # Choose the right model
    if example_type == ExampleType.CIRCLE:
        biorbd_model_path = "models/ArmModel_circle.bioMod"
    else:
        biorbd_model_path = "models/ArmModel_bar.bioMod"

    # Add the model
    model = pyorerun.BiorbdModel(biorbd_model_path)
    model.options.show_marker_labels = False
    model.options.show_center_of_mass_labels = False
    model.options.show_muscle_labels = False

    # Initialize the animation
    t_span = np.linspace(0, final_time, n_shooting + 1)
    viz = pyorerun.PhaseRerun(t_span)

    # Add experimental emg
    pyoemg = pyorerun.PyoMuscles(
        data=excitations_sol,
        muscle_names=list(model.muscle_names),
        mvc=np.ones((model.nb_muscles,)),
        colormap="viridis",
    )

    # Add the end effector as persistent marker
    marker_trajectories = pyorerun.MarkerTrajectories(marker_names=["end_effector"], nb_frames=None)

    # Add the kinematics
    viz.add_animated_model(
        model, q_sol, muscle_activations_intensity=pyoemg, marker_trajectories=marker_trajectories
    )

    # Play
    viz.rerun("OCP solution")

