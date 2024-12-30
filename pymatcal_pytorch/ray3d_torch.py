import torch
import numpy as np
import sys


def read_cuboids(filename):
    d_cpu = torch.device("cpu")
    cuboids = torch.tensor([], device=d_cpu)
    with np.load(filename) as f:
        crystal_cuboids = torch.from_numpy(f["crystal cuboids"])
        plate_cuboids = torch.from_numpy(f["plate cuboids"])
        return plate_cuboids, crystal_cuboids


def get_fov_voxel_centers(fov_nvoxels_xyz, fov_mmpvx_xyz, device: torch.device):
    gridx, gridy, gridz = torch.meshgrid(
        torch.arange(fov_nvoxels_xyz[0], device=device),
        torch.arange(fov_nvoxels_xyz[1], device=device),
        torch.arange(fov_nvoxels_xyz[2], device=device),
        indexing="ij",
    )
    grid_tensor = torch.cat((gridx, gridy, gridz), dim=-1).view(-1, 3)
    return grid_tensor * fov_mmpvx_xyz.view(1, 3).expand_as(grid_tensor)


def get_rays(
    pa_arr: torch.Tensor, pb_arr: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Get rays from array of points a and array of points b
    """
    npa = pa_arr.shape[0]
    npb = pb_arr.shape[0]
    pa_arr_expanded = pa_arr.to(device).unsqueeze(1).expand((-1, npb, -1))
    pb_arr_expanded = pb_arr.to(device).unsqueeze(0).expand((npa, -1, -1))
    return torch.stack((pa_arr_expanded, pb_arr_expanded), dim=2)


def raytrace_torch_old(
    pAs: torch.Tensor, pBs: torch.Tensor, cuboids: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Raytrace function for a batch of point As , point Bs and cuboids
    """
    n_rays = pAs.shape[0] * pBs.shape[0]
    n_cuboids = cuboids.shape[0]

    # rays_expanded = (
    #     rays.reshape(-1, 2, 3)
    #     .unsqueeze(1)
    #     .unsqueeze(2)
    #     .expand(-1, n_cuboids, 6, -1, -1)
    # )

    pAs_expanded = pAs.view(-1, 1, 1, 3).expand(-1, n_cuboids, 6, -1)
    e1 = -cuboids[:, [2, 2, 1, 1, 1, 1]]
    e2 = -cuboids[:, [3, 3, 3, 3, 2, 2]]

    e1 = e1.unsqueeze(0).expand(n_rays, -1, -1, -1)
    e2 = e2.unsqueeze(0).expand(n_rays, -1, -1, -1)
    verts = (
        (cuboids[:, 0] - torch.sum(cuboids[:, 1:], dim=-2) * 0.5)
        .clone()
        .view(-1, 1, 3)
        .expand(-1, 6, -1)
    )
    print("verts tensor storage size:", sys.getsizeof(verts.storage))
    verts[:, 1::2] += cuboids[:, 1:]
    verts = verts.view(1, -1, 6, 3).expand(n_rays, -1, -1, -1)
    v1p = verts - rays_expanded[:, :, :, 0]

    vec = (
        torch.clone(rays[:, 1] - rays[:, 0])
        .view(-1, 1, 1, 3)
        .expand(-1, n_cuboids, 6, -1)
    )
    # cross product of e_1 and e_2
    cp1 = torch.cross(e1, e2, dim=-1)

    # cross product of v and v_1 prime
    cp2 = torch.cross(vec, v1p, dim=-1)

    # denominator of the Cramer's rule
    det = torch.sum(vec * cp1, dim=-1)

    EPSILON = 1e-10

    # if determinant is near zero, there is
    # no solution to the set of equations
    # and the ray is parallel to the triangle
    intersect_rays_idx = torch.abs(det) >= EPSILON

    # inverse of the determinant
    det_inv = 1.0 / det[intersect_rays_idx]

    # mask the rays that intersect the faces of the cuboids
    cp1_masked = cp1[intersect_rays_idx]
    cp2_masked = cp2[intersect_rays_idx]
    v1p_masked = v1p[intersect_rays_idx]
    e1_masked = e1[intersect_rays_idx]
    e2_masked = e2[intersect_rays_idx]

    # prepare the parameters for the Cramer's rule
    t = torch.zeros_like(det, dtype=torch.float32, device=device)
    u = torch.zeros_like(det, dtype=torch.float32, device=device)
    v = torch.zeros_like(det, dtype=torch.float32, device=device)

    # solve the Cramer's rule
    t[intersect_rays_idx] = torch.sum(v1p_masked * cp1_masked, dim=-1) * det_inv
    u[intersect_rays_idx] = torch.sum(e2_masked * cp2_masked, dim=-1) * det_inv
    v[intersect_rays_idx] = -torch.sum(e1_masked * cp2_masked, dim=-1) * det_inv

    # check if the intersection point is inside
    # the quadrilateral and on the line segment
    condition = (
        (u >= EPSILON)
        * (1 - u >= EPSILON)
        * (v >= EPSILON)
        * (1 - v >= EPSILON)
        * (t >= EPSILON)
        * (1 - t >= EPSILON)
    )

    t[torch.logical_not(condition)] = 0
    # return torch.sort(t, dim=-1, descending=True)[0][:, :, :2]


def raytrace_torch(
    pAs: torch.Tensor, pBs: torch.Tensor, cuboids: torch.Tensor, device: torch.device
) -> torch.Tensor:

    n_pAs = pAs.shape[0]
    n_pBs = pBs.shape[0]
    n_cuboids = cuboids.shape[0]

    rays_pAs = pAs.view(-1, 1, 3).expand(-1, n_pBs, -1)
    rays_pBs = pBs.view(1, -1, 3).expand(n_pAs, -1, -1)

    e1 = -cuboids[:, [2, 2, 1, 1, 1, 1]]
    e2 = -cuboids[:, [3, 3, 3, 3, 2, 2]]

    # verts = torch.zeros_like(cuboids[:, 0])
    verts = (
        (cuboids[:, 0] - torch.sum(cuboids[:, 1:], dim=-2) * 0.5)
        .view(-1, 1, 3)
        .repeat(1, 6, 1)
    )

    verts[:, 1::2] += cuboids[:, 1:]
    verts = verts.view(1, 1, -1, 6, 3).expand(n_pAs, n_pBs, -1, -1, -1)

    # cross product of e_1 and e_2
    e1_expanded = e1.unsqueeze(0).unsqueeze(0).expand(n_pAs, n_pBs, -1, -1, -1)
    e2_expanded = e2.unsqueeze(0).unsqueeze(0).expand(n_pAs, n_pBs, -1, -1, -1)
    cp1 = (
        torch.cross(e1, e2, dim=-1)
        .view(1, 1, -1, 6, 3)
        .expand(n_pAs, n_pBs, -1, -1, -1)
    )
    vec = (
        (rays_pBs - rays_pAs)
        .view(n_pAs, n_pBs, 1, 1, 3)
        .expand(-1, -1, n_cuboids, 6, -1)
    )
    # denominator of the Cramer's rule
    det = torch.sum(vec * cp1, dim=-1)
    EPSILON = 1e-10

    # if determinant is near zero, there is
    # no solution to the set of equations
    # and the ray is parallel to the triangle
    intersect_rays_idx = torch.abs(det) >= EPSILON
    intersect_rays_idx = intersect_rays_idx.clone().view(n_pAs, n_pBs, n_cuboids, 6)

    # inverse of the determinant
    det_inv = det[intersect_rays_idx].clone()
    det_inv = 1.0 / det_inv

    del det
    v1p = (
        verts[intersect_rays_idx]
        - rays_pAs.view(n_pAs, n_pBs, 1, 1, 3).expand(-1, -1, n_cuboids, 6, -1)[
            intersect_rays_idx
        ]
    )

    # # cross product of v and v_1 prime
    cp2 = torch.cross(vec[intersect_rays_idx], v1p, dim=-1)

    # par is u in the Cramer's rule
    # prepare the parameters for the Cramer's rule
    par = torch.zeros((n_pAs, n_pBs, n_cuboids, 6), dtype=torch.float32, device=device)
    # solve the Cramer's rule
    par[intersect_rays_idx] = (
        torch.sum(e2_expanded[intersect_rays_idx] * cp2, dim=-1) * det_inv
    )
    condition = (par >= EPSILON) * (1 - par >= EPSILON)
    del par

    # par is v in the Cramer's rule
    # prepare the parameters for the Cramer's rule
    par = torch.zeros((n_pAs, n_pBs, n_cuboids, 6), dtype=torch.float32, device=device)
    # solve the Cramer's rule
    par[intersect_rays_idx] = (
        -torch.sum(e1_expanded[intersect_rays_idx] * cp2, dim=-1) * det_inv
    )
    condition *= (par >= EPSILON) * (1 - par >= EPSILON)
    del par

    # par is t in the Cramer's rule
    # prepare the parameters for the Cramer's rule
    par = torch.zeros((n_pAs, n_pBs, n_cuboids, 6), dtype=torch.float32, device=device)
    # solve the Cramer's rule
    par[intersect_rays_idx] = torch.sum(v1p * cp1[intersect_rays_idx], dim=-1) * det_inv
    condition *= (par >= EPSILON) * (1 - par >= EPSILON)

    par[torch.logical_not(condition)] = 0
    t = torch.sort(par, dim=-1, descending=True)[0][:, :, :, :2]
    del v1p, cp2, cp1, det_inv, condition

    # print(f"{'Shape of ts tensor:':64s}{t.shape}")
    return t


def get_subdiv_helper_tensor(n_subdivs, device: torch.device):
    lins_x = torch.linspace(0, 1, n_subdivs[0] + 1, device=device)
    lins_y = torch.linspace(0, 1, n_subdivs[1] + 1, device=device)
    lins_z = torch.linspace(0, 1, n_subdivs[2] + 1, device=device)
    c_x = (lins_x[:-1] + lins_x[1:]) * 0.5
    c_y = (lins_y[:-1] + lins_y[1:]) * 0.5
    c_z = (lins_z[:-1] + lins_z[1:]) * 0.5
    return (
        torch.stack(
            torch.meshgrid(c_x, c_y, c_z, indexing="ij"),
            dim=0,
        )
        .moveaxis(0, -1)
        .reshape(-1, 3)
        - 0.5
    )


def get_subdivs_centers(centers_xyz, sizes_xyz, n_subdivs_xyz, device: torch.device):
    subdiv_helper_xyz = get_subdiv_helper_tensor(n_subdivs_xyz, device=device)
    c_xyz_repeated = (
        centers_xyz.unsqueeze(1).repeat(1, torch.prod(n_subdivs_xyz), 1).reshape(-1, 3)
    )
    subdiv_helper_xyz_repeated = (
        subdiv_helper_xyz.unsqueeze(0).repeat(centers_xyz.shape[0], 1, 1).reshape(-1, 3)
    )
    if sizes_xyz.dim() == 1:
        sizes_xyz = sizes_xyz.unsqueeze(0).repeat(c_xyz_repeated.shape[0], 1)
    return c_xyz_repeated + subdiv_helper_xyz_repeated * sizes_xyz


def transform_points_batch(points, rots_xyz, device):
    # points: Nx3
    # rots_xyz: Nx3
    # device: torch.device
    # return: Nx3

    # bs = batch size
    bs = points.shape[0]
    cos_xyz = torch.cos(rots_xyz)
    sin_xyz = torch.sin(rots_xyz)
    # create rotation matrix
    r_matrice = torch.empty((bs, 3, 3), device=device)
    r_matrice[:, 0, 0] = cos_xyz[:, 1] * cos_xyz[:, 2]
    r_matrice[:, 0, 1] = cos_xyz[:, 1] * sin_xyz[:, 2]
    r_matrice[:, 0, 2] = -sin_xyz[:, 1]
    r_matrice[:, 1, 0] = (
        sin_xyz[:, 0] * sin_xyz[:, 1] * cos_xyz[:, 2] - cos_xyz[:, 0] * sin_xyz[:, 2]
    )
    r_matrice[:, 1, 1] = (
        sin_xyz[:, 0] * sin_xyz[:, 1] * sin_xyz[:, 2] + cos_xyz[:, 0] * cos_xyz[:, 2]
    )
    r_matrice[:, 1, 2] = sin_xyz[:, 0] * cos_xyz[:, 1]
    r_matrice[:, 2, 0] = (
        cos_xyz[:, 0] * sin_xyz[:, 1] * cos_xyz[:, 2] + sin_xyz[:, 0] * sin_xyz[:, 2]
    )
    r_matrice[:, 2, 1] = (
        cos_xyz[:, 0] * sin_xyz[:, 1] * sin_xyz[:, 2] - sin_xyz[:, 0] * cos_xyz[:, 2]
    )
    r_matrice[:, 2, 2] = cos_xyz[:, 0] * cos_xyz[:, 1]
    # rotate points
    return torch.bmm(r_matrice, points.unsqueeze(-1)).squeeze(-1)


def get_cuboids_vertices_batch(cuboids_centers_xyz, cuboids_vectors_xyz):
    cuboids_000 = cuboids_centers_xyz - torch.sum(cuboids_vectors_xyz, dim=1) * 0.5
    return torch.stack(
        [
            cuboids_000,  # 0
            cuboids_000 + cuboids_vectors_xyz[:, 0],  # 1
            cuboids_000 + cuboids_vectors_xyz[:, 1],  # 2
            cuboids_000 + cuboids_vectors_xyz[:, 2],  # 3
            cuboids_000 + cuboids_vectors_xyz[:, 0] + cuboids_vectors_xyz[:, 1],  # 4
            cuboids_000 + cuboids_vectors_xyz[:, 1] + cuboids_vectors_xyz[:, 2],  # 5
            cuboids_000 + cuboids_vectors_xyz[:, 2] + cuboids_vectors_xyz[:, 0],  # 6
            cuboids_000
            + cuboids_vectors_xyz[:, 0]
            + cuboids_vectors_xyz[:, 1]
            + cuboids_vectors_xyz[:, 2],  # 7
        ],
        dim=1,
    )


def get_faces_seqs_batch(n_cuboids, device):
    cuboids_faces_seqs = torch.tensor(
        [
            [0, 1, 4, 2],
            [3, 6, 7, 5],
            [0, 3, 5, 2],
            [1, 6, 7, 4],
            [0, 1, 6, 3],
            [2, 4, 7, 5],
        ],
        device=device,
    )

    faces_seqs_increment = (
        (torch.arange(n_cuboids, device=device) * 8)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, 6, 4)
    )
    return faces_seqs_increment + cuboids_faces_seqs.unsqueeze(0).repeat(
        n_cuboids, 1, 1
    )


def get_cuboids(geoms, cuboids_rots_xyz, rshift, device):
    panel_00_cuboids_centers_xyz = (geoms[:, :6:2] + geoms[:, 1:6:2]) / 2
    panel_00_cuboids_sizes_xyz = geoms[:, 1:6:2] - geoms[:, :6:2]
    ncuboids_panel_00 = panel_00_cuboids_centers_xyz.shape[0]
    nrots = cuboids_rots_xyz.shape[0]

    panel_00_cuboids_centers_xyz[:, 1] += -panel_00_cuboids_centers_xyz.mean(dim=0)[1]
    panel_00_cuboids_centers_xyz[:, 0] += rshift
    cuboids_centers_xyz = (
        panel_00_cuboids_centers_xyz.unsqueeze(0).repeat(nrots, 1, 1).to(device)
    )

    cuboids_centers_xyz = transform_points_batch(
        cuboids_centers_xyz.reshape(-1, 3), cuboids_rots_xyz.reshape(-1, 3), device
    ).reshape(nrots, -1, 3)
    panel_00_cuboids_vectors_xyz = torch.stack(
        [
            torch.stack(
                [
                    panel_00_cuboids_sizes_xyz[:, 0],
                    torch.zeros(ncuboids_panel_00, device=device),
                    torch.zeros(ncuboids_panel_00, device=device),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros(ncuboids_panel_00, device=device),
                    panel_00_cuboids_sizes_xyz[:, 1],
                    torch.zeros(ncuboids_panel_00, device=device),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.zeros(ncuboids_panel_00, device=device),
                    torch.zeros(ncuboids_panel_00, device=device),
                    panel_00_cuboids_sizes_xyz[:, 0],
                ],
                dim=-1,
            ),
        ],
        dim=1,
    )
    cuboids_vectors_xyz = panel_00_cuboids_vectors_xyz.repeat(nrots, 1, 1)
    cuboids_vectors_xyz = transform_points_batch(
        cuboids_vectors_xyz.reshape(-1, 3),
        cuboids_rots_xyz.unsqueeze(2).repeat(1, 1, 3, 1).reshape(-1, 3),
        device,
    ).reshape(nrots, -1, 3, 3)
    return cuboids_centers_xyz, cuboids_vectors_xyz
