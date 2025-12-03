import numpy as np
import casadi as cas


def get_bounds_and_init(
    model,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Get all the bounds and initial guesses for the states and controls.
    """
    # Optimized in Tom's version TODO remove
    shoulder_pos_initial = 0.349065850398866
    elbow_pos_initial = 2.245867726451909
    shoulder_pos_final = 0.959931088596881
    elbow_pos_final = 1.159394851847144


    n_muscles = model.nb_muscles
    n_q = model.nb_q
    n_k_fb = model.n_k_fb
    n_shooting = model.n_shooting

    # Q
    lbq = np.zeros((n_q, n_shooting + 1))
    ubq = np.zeros((n_q, n_shooting + 1))
    ubq[0, :] = np.pi / 2
    ubq[1, :] = 7 / 8 * np.pi
    q0 = np.zeros((n_q, n_shooting + 1))
    q0[0, :] = np.linspace(shoulder_pos_initial, shoulder_pos_final, n_shooting + 1)  # Shoulder
    q0[1, :] = np.linspace(elbow_pos_initial, elbow_pos_final, n_shooting + 1)  # Elbow

    # Qdot
    lbqdot = np.zeros((n_q, n_shooting + 1))
    lbqdot[:, 1:] = -10 * np.pi
    ubqdot = np.zeros((n_q, n_shooting + 1))
    ubqdot[:, 1:] = 10 * np.pi
    qdot0 = np.zeros((n_q, n_shooting + 1))

    # MuscleActivation
    lbmusa = np.ones((n_muscles, n_shooting + 1)) * 1e-6
    ubmusa = np.ones((n_muscles, n_shooting + 1))
    musa0 = np.ones((n_muscles, n_shooting + 1)) * 0.1

    states_lower_bounds = {
        "q": lbq,
        "qdot": lbqdot,
        "mus_activation": lbmusa,
    }
    states_upper_bounds = {
        "q": ubq,
        "qdot": ubqdot,
        "mus_activation": ubmusa,
    }
    states_initial_guesses = {
        "q": q0,
        "qdot": qdot0,
        "mus_activation": musa0,
    }

    # MuscleExcitation
    lbmuse = np.ones((n_muscles, n_shooting + 1)) * 1e-6
    ubmuse = np.ones((n_muscles, n_shooting + 1))
    muse0 = np.ones((n_muscles, n_shooting + 1)) * 0.1

    # K
    lbk = np.ones((n_k_fb, n_shooting + 1)) * -10
    ubk = np.ones((n_k_fb, n_shooting + 1)) * 10
    k0 = np.ones((n_k_fb, n_shooting + 1)) * 0.1

    controls_lower_bounds = {
        "mus_excitation": lbmuse,
        "k": lbk,
    }
    controls_upper_bounds = {
        "mus_excitation": ubmuse,
        "k": ubk,
    }
    controls_initial_guesses = {
        "mus_excitation": muse0,
        "k": k0,
    }

    return (
        states_lower_bounds,
        states_upper_bounds,
        states_initial_guesses,
        controls_lower_bounds,
        controls_upper_bounds,
        controls_initial_guesses,
    )

def get_noises_magnitude() -> tuple[np.ndarray, np.ndarray]:
    """
    Get the motor and sensory noise magnitude.
    """

    # TODO: see for the dt**2 thing

    n_q = 2
    motor_noise_std = 0.05  # Tau noise
    wPq_std = 0.0001  # Hand position noise
    wPqdot_std = 0.0001  # Hand velocity noise

    motor_noise_magnitude = cas.DM(np.array([motor_noise_std] * n_q))
    sensory_noise_magnitude = cas.DM(
        np.array(
            [
                wPq_std,
                wPq_std,
                wPqdot_std,
                wPqdot_std,
            ]
        )
    )

    return motor_noise_magnitude, sensory_noise_magnitude