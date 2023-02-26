import time
import torch
import argparse
import numpy as np
from hydra import initialize
from omegaconf import OmegaConf
from h_tsp import (
    readDataFile,
    HTSP_PPO,
    utils,
    VecEnv,
    RLSolver,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower_model", type=str, help="Path to the lower level model checkpoint"
    )
    parser.add_argument(
        "--upper_model", type=str, help="Path to the upper level model checkpoint"
    )
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--graph_size", type=int, default=1000)
    parser.add_argument("--frag_len", type=int, default=200, help="Sub-problem size")
    parser.add_argument(
        "--max_new_cities",
        type=int,
        default=190,
        help="Maximum number of new cities in sub-problem",
    )
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--data_augment", default=False, action="store_true")
    parser.add_argument(
        "--improvement_step", type=int, default=0, help="Number of improvement steps"
    )
    parser.add_argument("--time_limit", type=float, default=100.0, help="Time limit")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Evaluate batch size"
    )
    return parser.parse_args()


def main(args):
    graph_size = args.graph_size
    frag_len = args.frag_len
    max_new_cities = args.max_new_cities
    k = args.k
    bsz = args.batch_size

    ckpt = torch.load(args.upper_model)
    with initialize(config_path=".", version_base="1.1"):
        cfg = OmegaConf.create(ckpt["hyper_parameters"])

    cfg.low_level_load_path = args.lower_model

    model = HTSP_PPO(cfg).cuda()
    model.load_state_dict(ckpt["state_dict"])
    rl_solver = RLSolver(model.low_level_model, frag_len)

    data_file = f"data/cluster/tsp{graph_size}_test_concorde.txt"
    data = readDataFile(data_file)
    sample_nums = data.shape[0]
    if args.data_augment:
        data = utils.augment_xy_data_by_8_fold(data)
    print(f"{data.shape=}")

    vec_env = VecEnv(
        k=k, frag_len=frag_len, max_new_nodes=max_new_cities, max_improvement_step=0
    )
    results = np.array([])
    start_t = time.time()
    for i in range(0, data.shape[0], bsz):
        batch_start = time.time()
        batch_time_limit = args.time_limit * bsz
        batch_data = data[i : i + bsz]
        print(f"{i}/{batch_data.shape[0]}")
        s = vec_env.reset(batch_data.to(model.device))
        while not vec_env.done:
            a = model(s).detach()
            # random action for comparison
            # a = vec_env.random_action().to(model.device)
            s, r, d, info = vec_env.step(
                a, rl_solver, frag_buffer=model.val_frag_buffer
            )
        print(np.array([e.state.current_tour_len.item() for e in vec_env.envs]).mean())
        if args.improvement_step > 0:
            for env in vec_env.envs:
                env.max_improvement_step = args.improvement_step
            while not vec_env.done:
                if time.time() - batch_start > batch_time_limit:
                    break
                a = vec_env.random_action().to(model.device)
                s, r, d, info = vec_env.step(
                    a, rl_solver, frag_buffer=model.val_frag_buffer
                )
        length = np.array([e.state.current_tour_len.item() for e in vec_env.envs])
        results = np.concatenate((results, length))

    duration = time.time() - start_t

    if args.data_augment:
        results = results.reshape(8, -1).min(axis=0)

    assert results.shape[0] == sample_nums, f"{length.shape[0]=}, {sample_nums=}"

    return duration, results.mean()


if __name__ == "__main__":
    args = parse_args()
    durations = []
    lengths = []
    for i in range(args.repeat_times):
        duration, length = main(args)
        durations.append(duration)
        lengths.append(length)
    print(f"average duration: {np.average(durations)}")
    print(f"average length: {np.average(lengths)}")
