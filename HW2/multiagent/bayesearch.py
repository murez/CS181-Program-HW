from bayes_opt import BayesianOptimization
import os
import json
import subprocess
import pacman


def score(score_value, ghost_state_base, ghost_state_zero, ghost_value, food_num_value, food_dis_value,
          capsule_dis_value,
          capsule_dis_num):
    data = {}
    data["score_value"] = score_value
    data["ghost_state_base"] = ghost_state_base
    data["ghost_state_zero"] = ghost_state_zero
    data["ghost_value"] = ghost_value
    data["food_num_value"] = food_num_value
    data["food_dis_value"] = food_dis_value
    data["capsule_dis_value"] = capsule_dis_value
    data["capsule_dis_num"] = capsule_dis_num
    with open("./data_2.json", "w") as f:
        json.dump(data, f)

    print("start with data:")
    print(data)
    args = pacman.readCommand(['-l', 'contestClassic', '-p', 'ContestAgent', '-g', 'DirectionalGhost', '-q', '-n', '5'])
    pacman.runGames(**args)
    with open("average_2.txt", "r") as f:
        rs = float(f.readline())
    print("final score: ", rs)
    data['final'] = rs

    with open("./learned_2.txt",'a+') as f:
        f.write(data.__str__())
        f.write("\n")
    return rs


pbounds = {
    "score_value": (0.01, 10.0),
    "ghost_state_base": (0.01, 100.0),
    "ghost_state_zero": (-20.0, 20.0),
    "ghost_value": (-10.0, 10.0),
    "food_num_value": (-10.0, 10.0),
    "food_dis_value": (-10.0, 10.0),
    "capsule_dis_value": (-10.0, 10.0),
    "capsule_dis_num": (-10.0, 50.0)
}

optimizer = BayesianOptimization(
    f=score,
    pbounds=pbounds,
    random_state=10,
)
optimizer.maximize(n_iter=1000)
