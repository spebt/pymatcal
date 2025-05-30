import os

import torch.distributed as dist
from torch import Tensor, arange, empty, empty_like
from torch import float32 as torch_float32
from torch import ones, tensor,cat,stack


def setup_distributed():
    # print("Setting up distributed environment...")
    dist.init_process_group(backend="gloo", init_method="env://")
    # print("Done setup distributed environment...")


def cleanup_distributed(rank: int):
    dist.destroy_process_group()
    # print(f"Rank {rank}: Process group destroyed.")


def run_gather(
    rank: int, multiplier, local_tensor: Tensor, gathered_tensors=None
):
    """
    A function to run on each process to demonstrate gather.
    """

    local_tensor.copy_(ones(8, dtype=torch_float32) * rank * multiplier)

    dist.gather(local_tensor, gather_list=gathered_tensors, dst=0)


def main():
    """
    Main function to run the gather operation in a distributed setting.
    """

    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a tensor specific to each rank
    local_tensor = empty(8, dtype=torch_float32)
    if rank == 0:
        # Prepare a list to gather tensors from all ranks
        gathered_tensors = [empty_like(local_tensor) for _ in range(world_size)]
    else:
        gathered_tensors = None
    
    out_str = "Final gathered tensor: " if rank == 0 else ""
    for multiplier in arange(0, 100) * 0.1:
        # Run the gather operation
        run_gather(rank, multiplier, local_tensor, gathered_tensors)
        if rank == 0:
            # out_str += f"Multiplier: {multiplier.item()}, Gathered Tensors: {gathered_tensors}\n"
            summation = stack(gathered_tensors).sum(dim=0)
            out_str += f"Summation: {summation.tolist()}\n"
    #     if rank == 0:
    #         out_str += f"{summation.tolist()}\n"

    # # Cleanup the distributed environment
    if rank == 0:
        print(out_str)
    # else:
    #     print(f"Rank {rank} completed its work.")
    dist.barrier()  # Ensure all ranks complete before cleanup
    cleanup_distributed(rank)


if __name__ == "__main__":
    main()
