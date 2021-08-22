import optuna
import numba


def objective(trial):
    x = trial.suggest_float("x", -7, 7)  # Name and range
    y = trial.suggest_float("y", -7, 7)
    return (x - 1) ** 2 + (y + 3) ** 2


study = optuna.create_study()
study.optimize(objective, n_trials=100)  # number of iterations

print(study.best_params)  # return best parameters

study.optimize(objective, n_trials=100)  # apply optimization for another 100

print(study.best_params)

# direction="minimize" for loss
study = optuna.create_study(direction="maximize")  # Direction depends on metric


def objective(trial):
    params = {
        "": trial.suggest_float("", 0, 0, step=0)
    }
