from git import Repo
import os


repo = Repo(os.path.dirname(__file__), search_parent_directories=True)
root_path = repo.git.rev_parse("--show-toplevel")


def reducing(x, y):
    if len(str(x)) == 19:
        return str(x)[0:10] + ',' + str(y)[0:10]
    else:
        return str(x) + ',' + str(y)[0:10]


__all__ = ["root_path"]
