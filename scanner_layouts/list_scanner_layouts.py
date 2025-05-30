if __name__ == "__main__":
    import sys
    import os
    from torch import load as torch_load
    from helper import print_keys

    filename = sys.argv[1]
    # Check if the file exists
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        raise FileNotFoundError(f"File {filename} does not exist.")
    # Load the file
    scanner_layouts_data = torch_load(filename)

    # Print the keys in the dictionary
    print_keys(scanner_layouts_data)
