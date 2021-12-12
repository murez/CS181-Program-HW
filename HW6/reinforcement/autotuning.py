import json
from bayes_opt import BayesianOptimization
import os


def run_pacman(bias,
               closest_weak_ghost_0,
               closest_weak_ghost_1,
               strong_ghost_1_step_0,
               strong_ghost_1_step_1,
               strong_ghost_2_step_0,
               strong_ghost_2_step_1,
               weak_ghost_0,
               weak_ghost_1,
               eats_capsule,
               eats_food,
               closest_capsule_0,
               closest_capsule_1,
               closest_food_0,
               closest_food_1
               ):
    parameters = {
        "bias": bias,
        "closest_weak_ghost_0": closest_weak_ghost_0,
        "closest_weak_ghost_1": closest_weak_ghost_1,
        "strong_ghost_1_step_0": strong_ghost_1_step_0,
        "strong_ghost_1_step_1": strong_ghost_1_step_1,
        "strong_ghost_2_step_0": strong_ghost_2_step_0,
        "strong_ghost_2_step_1": strong_ghost_2_step_1,
        "weak_ghost_0": weak_ghost_0,
        "weak_ghost_1": weak_ghost_1,
        "eats_capsule": eats_capsule,
        "eats_food": eats_food,
        "closest_capsule_0": closest_capsule_0,
        "closest_capsule_1": closest_capsule_1,
        "closest_food_0": closest_food_0,
        "closest_food_1": closest_food_1,
    }
    json.dump(parameters, open("parameters.json", "w"))
    # os.system("python autograder.py -q q11")

    output = os.popen("python autograder.py -q q11").read()

    score = float(output.split("Average Score:")[-1].split("\n")[0])
    print(parameters)
    print(score)
    return score


pbounds = {
    "bias": (-3, 3),
    "closest_weak_ghost_0": (-10, 10),
    "closest_weak_ghost_1": (-7, -5),
    "strong_ghost_1_step_0": (-8, -4),
    "strong_ghost_1_step_1": (-10, 0),
    "strong_ghost_2_step_0": (-10, 5),
    "strong_ghost_2_step_1": (-5, 5),
    "weak_ghost_0": (-5, 5),
    "weak_ghost_1": (-5, 10),
    "eats_capsule": (0, 10),
    "eats_food": (0, 10),
    "closest_capsule_0": (-5, 10),
    "closest_capsule_1": (-1, 5),
    "closest_food_0": (0, 10),
    "closest_food_1": (-4, 2)
}

optimizer = BayesianOptimization(
    f=run_pacman,
    pbounds=pbounds,
    random_state=4,
)

optimizer.maximize(
    init_points=500,
    n_iter=6000,
)
