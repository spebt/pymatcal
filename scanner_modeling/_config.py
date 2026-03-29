from torch import float32

#: Global floating-point dtype for all 3D geometry / raytracing tensors.
DTYPE = float32

#: Canonical distance unit used throughout geometry/raytracer.
#: All distances passed into the 3D pipeline must be in these units.
DISTANCE_UNIT = "mm"

RAY_CHUNK_SIZE = 50_000  # Safe default for Phase 5

__all__ = ["DTYPE", "DISTANCE_UNIT"]
