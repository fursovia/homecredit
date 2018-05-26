"""Перебор гиперпараметров"""

import argparse
import os
from subprocess import check_call
import sys

from model.utils import Params


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='/data/i.fursov/experiments',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")



def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}"
    cmd = cmd.format(python=PYTHON, model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    # margins = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # betas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    #
    # for m in margins:
    #     for b in betas:
    #         if b >= m:
    #         # Modify the relevant parameter in params
    #             params.margin = m
    #             params.beta = b
    #             # Launch job (name has to be unique)
    #             job_name = "m_{0}_b_{1}".format(m, b)
    #             launch_training_job(args.parent_dir, args.data_dir, job_name, params)

    betas = [0.05, 0.3, 0.5, 0.6, 0.7]
    margins = [0.05, 0.1, 0.2, 0.5]

    for b in betas:
        for m in margins:
            params.beta = b
            params.margin = m
            # Launch job (name has to be unique)
            job_name = "beta_{0}_margin{1}".format(b, m)
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)