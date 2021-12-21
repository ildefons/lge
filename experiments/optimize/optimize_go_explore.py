import gym
import numpy as np
import optuna
from go_explore.go_explore.cell_computers import PandaObjectCellComputer
from go_explore.go_explore.go_explore import GoExplore


def objective(trial: optuna.Study):
    count_pow = trial.suggest_categorical("count_pow", [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    subgoal_horizon = trial.suggest_categorical("subgoal_horizon", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    done_delay = trial.suggest_categorical("done_delay", [0, 1, 2, 4, 8, 16, 32, 64])

    results = []

    for _ in range(5):
        env = gym.make("PandaNoTask-v0", nb_objects=1)
        ge = GoExplore(env, PandaObjectCellComputer(), subgoal_horizon, done_delay, count_pow)
        ge.exploration(50000)
        results.append(ge.archive.nb_cells)

    return np.median(results)


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///example.db",
        direction="maximize",
        study_name="PandaObjectGoExplore",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    importances = optuna.importance.get_param_importances(study)
    print(importances)
