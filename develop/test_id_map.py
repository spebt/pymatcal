from torch import Tensor, tensor
import math

def make_ids_map(n_sfov: int, n_crystals: int) -> Tensor:
    return tensor([(i, j) for i in range(n_sfov) for j in range(n_crystals)])


def run(rank, size, id_map: Tensor, *args):
    
    local_id_map = id_map[rank::size]
    print(
        f"Rank {rank} of {size}, local id map size: {local_id_map}"
    )


if __name__ == "__main__":
    # Example usage

    size = 128
    id_map = make_ids_map(10, 13)
    for rank in range(size):
        run(rank, size, id_map)
