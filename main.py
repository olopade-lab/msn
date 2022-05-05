# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from pathlib import Path
from maicara.preprocessing.utils import log_code_state

import torch
import torch.multiprocessing as mp

from pprint import pformat
import yaml


from src.msn_train import main as msn

from src.utils import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname", type=str, help="name of config file to load", default="configs.yaml"
)
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=None,
    help="which devices to use on local machine",
)


def process_main(rank, fname, world_size, devices):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # -- load script params
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")
        logger.info(pformat(params))

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    if rank == 0:
        dump = os.path.join(
            params["logging"]["folder"],
            params["logging"]["tag"],
            "params-msn-train.yaml",
        )
        Path(os.path.dirname(dump)).mkdir(parents=True, exist_ok=True)
        with open(dump, "w+") as f:
            yaml.dump(params, f)
        log_code_state(os.path.dirname(dump))

    return msn(params)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.devices is None:
        args.devices = [f"cuda:{i}" for i in torch.cuda.device_count()]

    num_gpus = len(args.devices)
    mp.spawn(process_main, nprocs=num_gpus, args=(args.fname, num_gpus, args.devices))
