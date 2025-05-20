if __name__ == "__main__":
    import os
    import sys

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import ScalarFormatter
    from torch import Tensor
    from torch import load as torch_load
    from torch import tensor, zeros

    # Define the color stops
    ppdf_colors = [
        (1, 1, 1),  # white
        (1, 1, 0),  # yellow
        (1, 0.65, 0),  # orange
        (1, 0, 0),  # red
    ]
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list("ppdf_hot", ppdf_colors)

    # Set the colorbar format
    formatter = ScalarFormatter()
    formatter.set_powerlimits((-2, 2))

    # Get the input tensor filenames from command line arguments
    input_tensor_filenames = sys.argv[1:]

    # Check if all files exist
    for input_tensor_filename in input_tensor_filenames:
        if not os.path.exists(input_tensor_filename):
            print(f"File not found: {input_tensor_filename}")
            sys.exit(1)

    # Create a figure with a constrained layout
    fig = plt.figure(layout="constrained", figsize=(9.5, 8))
    # Add empty imshow to the figure
    imshow_obj = plt.imshow(zeros((64, 64)).numpy(), cmap=cmap)
    colorbar_obj = plt.colorbar(
        imshow_obj,
        format=formatter,
    )

    for input_tensor_filename in input_tensor_filenames:
        # Load the tensor from the file
        ppdf_data: Tensor = torch_load(input_tensor_filename)
        ppdf_size_one_dimension = tensor(ppdf_data.shape[0]).sqrt()
        max_value = ppdf_data.max()
        min_value = ppdf_data.min()
        print(
            f"Max value: {max_value}, Min value: {min_value}, Shape: {ppdf_data.shape}"
        )
        new_data = ppdf_data.view(
            int(ppdf_size_one_dimension), int(ppdf_size_one_dimension)
        ).T.numpy()
        imshow_obj.set_data(new_data)
        imshow_obj.set_clim(min_value.item(), max_value.item())
        colorbar_obj.update_normal(imshow_obj)
        fig.canvas.draw()
        plt.title("PPDF of " + input_tensor_filename)
        plt.xlabel("pixel x-index")
        plt.ylabel("pixel y-index")
        plt.savefig(input_tensor_filename.split(".")[0] + ".png")
