# Scanner Layouts

This directory contains the layout files for various scanners. Each layout file is `Torch` file that can be loaded with the `torch.load` function. The layouts are used to define the geometry of the scanners at different positions.

## Layout Files

- Each file in this directory (with the `.tensor` extension) contains a `Dict` representing the scanner layouts at different positions.
- The filename typically encodes a unique identifier for the scanner configuration.

## Usage

To load a scanner layout in Python:

```python
import torch
layouts = torch.load("scanner_layouts/scanner_layouts_77faff53af5863ca146878c7c496c75e.tensor")
```

## File Naming Convention

- Files are named as `scanner_layouts_<hash>.tensor`, where `<hash>` is a unique identifier for the layout.
- This allows for easy referencing and prevents naming conflicts.
- The hash generation uses the `hashlib` library to calculate MD5 hash of the layout data, ensuring uniqueness.

## List of Available Layouts

> [!NOTE]
> Translations = 1 means the scanner field of view centered at the origin. No translation is applied.

You can take a quick look of the scanner layouts with `list_scanner_layouts.py` script in the directory

```bash
python list_scanner_layouts.py <layouts_filename>
```

|             MDF5 Hash              |                 Description                | Rotations | Translations |
|------------------------------------|--------------------------------------------|-----------|--------------|
| `77faff53af5863ca146878c7c496c75e` | Layouts from a randomly generated scanner  |     24    |      1       |
| `e1531c3444e51439add2f18f5714fc50` | Layouts from ordered scanner               |     24    |      1       |
