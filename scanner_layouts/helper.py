import hashlib
from torch import Tensor, tensor



def generate_md5_from_tensors(*tensors: Tensor) -> str:
    hash_obj = hashlib.md5()
    for tensor in tensors:
        hash_obj.update(tensor.numpy().tobytes())
    return hash_obj.hexdigest()

# recursively print all the keys in the dictionary
def print_keys(d, parent_key=None, lvl=0):
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, Tensor):
            tensor_info = (
                f"Shape {tuple(tensor(v.shape).tolist())}"
                if v.numel() > 6
                else f"{v.tolist()}"
            )
            print(f"{'  ' * lvl}{k}: {tensor_info}")
        if isinstance(v, dict):
            print(f"{'  ' * lvl}{k}:")
            print_keys(v, new_key, lvl + 1)
        if isinstance(v, str):
                print(f"{'  ' * lvl}{k}: {v}")
