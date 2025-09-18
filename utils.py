import git
from datetime import date

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
