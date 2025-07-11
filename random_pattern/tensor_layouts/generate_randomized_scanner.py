"""
Generate a randomized scanner layout with given parameters.


"""

if __name__ == "__main__":
    import sys
    import os

    # import torch
    from torch import tensor, save as torch_save
    from helper import (
        rotate_vertices_2d_batch,
        scanner_layout_random_last_full,
        # generate_sha256_from_tensors,
        generate_md5_from_tensors,
    )

    plate_segments, detector_units = scanner_layout_random_last_full(
        tensor([0.5, 0.4, 93, 38])
    )

    unique_id = generate_md5_from_tensors(detector_units, plate_segments)

    out_file_name = f"scanner_{unique_id}.tensor"
    print(f"Save to:\n  {out_file_name}")
    torch_save(
        {
            "detector units": detector_units,
            "plate segments": plate_segments,
        },
        out_file_name,
    )
