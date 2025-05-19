import torch
import torch.distributed as dist
import os

def setup_distributed():
    print("Setting up distributed environment...")
    dist.init_process_group(backend='gloo', init_method='env://')
    print("Done setup distributed environment...")
def cleanup_distributed():
    dist.destroy_process_group()

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    elif rank == 1:
        dist.recv(tensor=tensor, src=0)
    print(f'Rank {rank} has data {tensor[0]}')

def main():
    setup_distributed()
    rank = dist.get_rank()
    size = dist.get_world_size()
    run(rank, size)
    cleanup_distributed()

if __name__ == "__main__":
    main()