import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Configuration ---
# Set the path to the .npz file you created
base_dir = "/vscratch/grp-rutaoyao/Harsh/ALL_CONFIGS/random_pattern/reconstruction/data"
npz_file_path = os.path.join(base_dir, "recon_mlem_torch_derenzo.npz")
output_dir = "./reconstruction_images" # Directory to save output images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Will save output images to: {output_dir}")

# --- 2. Load and Inspect the Data ---
try:
    data = np.load(npz_file_path)
except FileNotFoundError:
    print(f"ERROR: The file was not found at {npz_file_path}")
    print("Please make sure the path is correct and the MLEM script has run successfully.")
    exit()

# See what arrays are stored in the file
print(f"Keys found in the .npz file: {data.files}")

# The 'estimates' key holds our reconstructed images
reconstructions = data['estimates']

# The shape will be (number_of_saved_iterations, height, width)
print(f"Shape of reconstructions array: {reconstructions.shape}")

# --- 3. Display the FINAL Reconstructed Image ---
# The last image in the array is the final result
final_recon = reconstructions[-1]

plt.figure(figsize=(8, 8))
plt.imshow(final_recon, cmap='gray')
plt.colorbar(label="Image Intensity")
plt.title("Final Reconstructed Image")
plt.xlabel("Pixel")
plt.ylabel("Pixel")
# Save the figure
final_image_path = os.path.join(output_dir, "final_reconstruction.png")
plt.savefig(final_image_path, dpi=300, bbox_inches='tight')
print(f"Saved final reconstruction to: {final_image_path}")
plt.show() # This will pop up a window with the image

# --- 4. Display Images from Different Iterations ---
# Let's compare the first, middle, and last saved estimates
num_saved_estimates = reconstructions.shape[0]
indices_to_show = [0, num_saved_estimates // 2, -1] # First, middle, last

# The MLEM script saved every 5th iteration
iteration_numbers = [idx * 5 for idx in indices_to_show]

fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # 1 row, 3 columns
fig.suptitle("Reconstruction Progress", fontsize=16)

for i, (ax, idx) in enumerate(zip(axes, indices_to_show)):
    ax.imshow(reconstructions[idx], cmap='gray')
    ax.set_title(f"After ~Iteration {iteration_numbers[i]}")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

# Save the comparison figure
comparison_image_path = os.path.join(output_dir, "comparison_reconstruction.png")
plt.savefig(comparison_image_path, dpi=300, bbox_inches='tight')
print(f"Saved comparison image to: {comparison_image_path}")
plt.show()

# --- 5. (Bonus) Create an Animated GIF of the Reconstruction ---
# This requires the 'Pillow' library. If you don't have it, run:
# pip install Pillow
print("\nAttempting to create an animation...")
try:
    from matplotlib.animation import FuncAnimation

    fig_anim, ax_anim = plt.subplots(figsize=(7, 7))
    plt.close(fig_anim) # Prevent the static figure from showing up now

    # Initialize the plot with the first frame
    im = ax_anim.imshow(reconstructions[0], cmap='gray', animated=True)
    ax_anim.set_xlabel("Pixel")
    ax_anim.set_ylabel("Pixel")
    cb = fig_anim.colorbar(im, ax=ax_anim, label="Image Intensity")
    
    def update(frame):
        """The function that draws each frame of the animation."""
        image_data = reconstructions[frame]
        im.set_array(image_data)
        im.set_clim(vmin=image_data.min(), vmax=image_data.max()) # Update colorbar limits
        ax_anim.set_title(f"Reconstruction after ~Iteration {frame * 5}")
        return im,

    # Create the animation object
    ani = FuncAnimation(
        fig=fig_anim,
        func=update,
        frames=len(reconstructions), # Number of frames to run
        interval=100, # Delay between frames in milliseconds
        blit=True
    )

    # Save the animation as a GIF
    gif_path = os.path.join(output_dir, "reconstruction_progress.gif")
    ani.save(gif_path, writer='pillow', fps=10) # fps = frames per second
    print(f"Successfully saved animation to: {gif_path}")

except ImportError:
    print("Could not import FuncAnimation. Skipping animation.")
    print("To create animations, please install required libraries.")
except Exception as e:
    print(f"An error occurred during animation creation: {e}")