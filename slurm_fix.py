"""Utils that facilitate training and evaluation

Source: https://github.mit.edu/ai4bio/mlworkflow/blob/master/mlworkflow/utils.py
"""

import os
import socket
import sys


def parse_node(node):
    """
    Parses a slurm node into a readable format for pytorch-lightning
    """
    prefix = node[:node.find("[")]
    node_range = node[node.find("[")+1:node.find("]")]
    split_range = node_range.split("-")
    digits = len(split_range[0])
    split_range = [int(i) for i in split_range]
    split_range = [str(i) for i in range(split_range[0], split_range[1]+1)]
    split_range_ = []
    for s in split_range:
        if len(s) < digits:
            s = '0'*(digits - len(s)) + s
        split_range_.append(s)
    split_range = split_range_

    parsed_node = [prefix+i for i in split_range]
    return parsed_node


def parse_nodelist(nodelist):
    """
    Parses a list of slurm nodes into a readable format for pytorch-lightning
    """
    split_nodelist = nodelist.split(",")

    parsed_nodelist = []
    for node in split_nodelist:

        if "[" in node and "]" in node:
            node = parse_node(node)

        if type(node) is list:
            parsed_nodelist = parsed_nodelist + node
        else:
            parsed_nodelist.append(node)

    return parsed_nodelist


def parse_env4lightning(verbose=False, nccl_debug=None):
    """
    Modified from Raiden here: https://github.mit.edu/mit-ai-accelerator/raiden/blob/master/raiden/utils.py#L40

    The following code is mostly designed to work around
    some issues with Lightning and non-standard slurm configurations
    """

    
    if os.environ.get("PARSE_4_LIGHTNING", "0") == "1":
        return

    os.environ["HYDRA_FULL_ERROR"] = "1"
    if nccl_debug is not None:
        os.environ["NCCL_DEBUG"] = nccl_debug

    # fix GPU device ids.
    # Reason: CUDA is labeled with gpu names rather than ordinals
    if 'CUDA_VISIBLE_DEVICES' in os.environ:

        if verbose:
            print("# CUDA_VISIBLE_DEVICES before processing: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]), file=sys.stderr)

        num_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',').__len__()
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(num_gpus))

        if verbose:
            print("# CUDA_VISIBLE_DEVICES after processing: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]), file=sys.stderr)
    elif verbose:
        print("# CUDA_VISIBLE_DEVICES not found.", file=sys.stderr)
    

    # Fix nodelist
    # Reason: llgrid combines consecutive hostnames into 1
    if "SLURM_NODELIST" in os.environ:
        named_nodelist = parse_nodelist(os.environ["SLURM_NODELIST"])
        os.environ["SLURM_NODELIST"] = " ".join(named_nodelist)

        if verbose:
            print("# SLURM_NODELIST: {}".format(os.environ["SLURM_NODELIST"]), file=sys.stderr)

        os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"].split(" ")[0]
    else:
        if verbose:
            print("# SLURM_NODELIST not found.", file=sys.stderr)
    #os.environ["MASTER_PORT"] = "30480"
    os.environ["PARSE_4_LIGHTNING"] = "1"
    """
    # Fix interactive node submissions
    # Reason: Ax will sometimes try to acquire the previous port after multiple sequential runs
    if "SUBMITIT_EXECUTOR" not in os.environ:
        print("# PORT MANUALLY ASSIGNED", file=sys.stderr)
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        os.environ["MASTER_PORT"] = str(port)
        print("# PORT ASSIGNMENT: {}".format(str(port)), file=sys.stderr)
    else:
        print("# PORT ASSIGNED THROUGH PL", file=sys.stderr)

    if verbose:
        print("# HOSTNAME: {}".format(socket.gethostname()), file=sys.stderr)

    # Set flag so this won't be repeated
    os.environ["PARSE_4_LIGHTNING"] = "1"
    """

def shutdown_lightning():
    os.environ["PARSE_4_LIGHTNING"] = "0"
