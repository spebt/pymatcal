if __name__ == "__main__":
    import sys
    import os
    from torch import (
        load as torch_load,
        tensor,
        Tensor,
    )
    from helper import (
        generate_md5_from_tensors,
    )

    filename = sys.argv[1]
    # Check if the file exists
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        raise FileNotFoundError(f"File {filename} does not exist.")
    # Load the file
    scanner_layouts_data = torch_load(filename)

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

    # Print the keys in the dictionary
    print_keys(scanner_layouts_data)
