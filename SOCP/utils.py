import git
from datetime import date
import casadi as cas

from enum import Enum


class ExampleType(Enum):
    """
    Selection of the type of example to solve
    """

    CIRCLE = "CIRCLE"
    BAR = "BAR"


def get_git_version():

    # Save the version of bioptim and the date of the optimization for future reference
    repo = git.Repo(search_parent_directories=True)
    commit_id = str(repo.commit())
    branch = str(repo.active_branch)
    bioptim_version = repo.git.version_info
    git_date = repo.git.log("-1", "--format=%cd")
    version_dic = {
        "commit_id": commit_id,
        "git_date": git_date,
        "branch": branch,
        "bioptim_version": bioptim_version,
        "date_of_the_optimization": date.today().strftime("%b-%d-%Y-%H-%M-%S"),
    }
    return version_dic
    
    
def RK4(x_prev, u, dt, motor_noise, forward_dyn_func, n_steps=5):
    h = dt / n_steps
    x_all = cas.DM.zeros((n_steps + 1, x_prev.shape[0]))
    x_all[0, :] = x_prev
    for i_step in range(n_steps):
        k1 = forward_dyn_func(
            x_prev,
            u,
            motor_noise,
        )
        k2 = forward_dyn_func(
            x_prev + h / 2 * k1,
            u,
            motor_noise,
        )
        k3 = forward_dyn_func(
            x_prev + h / 2 * k2,
            u,
            motor_noise,
        )
        k4 = forward_dyn_func(
            x_prev + h * k3,
            u,
            motor_noise,
        )

        x_all[i_step + 1, :] = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_prev = x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_all
