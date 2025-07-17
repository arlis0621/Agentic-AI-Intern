import os
import numpy as np
import trimesh

# --------------------------------------------------------------------------------
# Shape-Signature Tensor (SST) Creation
# --------------------------------------------------------------------------------
# This module implements the SST pipeline as described in ShipHullGAN:
# 1) Normalize hull mesh by length
# 2) Sample E=56 non-uniform transverse stations along [0,1]
# 3) Extract each cross-section:
#    - For xi < 0.1 (bulbous bow region), select the largest loop from transverse slicing to get a single simply-connected curve
#    - For xi >= 0.1, use standard transverse slicing
# 4) Uniformly sample N=25 vertical levels per section to build half-breadth vs. vertical matrices
# 5) Assemble geometry matrices X, Y, Z of shape (N, E)
# 6) Compute central, scale-invariant 3D moment invariants up to 4th order (35 values)


def load_and_normalize_mesh(stl_path):
    """
    Load an STL mesh, translate so x∈[0,1] and center Y,Z, then scale by original length L.
    Returns:
      mesh      : trimesh.Trimesh (vertices modified in-place)
      (L, B, D) : original bounding-box dimensions
    """
    mesh = trimesh.load_mesh(stl_path, process=True)
    minb, maxb = mesh.bounds  # (2,3)
    L, B, D = (maxb - minb)
    # translate: x-min->0, center Y & Z
    mesh.vertices -= np.array([
        minb[0],
        (minb[1] + maxb[1]) / 2,
        (minb[2] + maxb[2]) / 2
    ])
    # normalize by length
    mesh.vertices /= L
    return mesh, (L, B, D)


def get_stations():
    """
    Non-uniform station fractions along normalized hull length [0,1]:
      5 in [0.0,0.1), 10 in [0.1,0.3), 35 in [0.3,0.8), 6 in [0.8,1.0]
    Returns array shape (56,)
    """
    r1 = np.linspace(0.00, 0.10,  5, endpoint=False)
    r2 = np.linspace(0.10, 0.30, 10, endpoint=False)
    r3 = np.linspace(0.30, 0.80, 35, endpoint=False)
    r4 = np.linspace(0.80, 1.00,  6, endpoint=True)
    return np.concatenate([r1, r2, r3, r4])


def sample_section(mesh, xi, N=25):
    """
    Standard transverse slicing at x=xi plane.
    Returns array (N,2): [half_breadth, z] pairs.
    """
    section = mesh.section(
        plane_origin=[xi, 0, 0],
        plane_normal=[1, 0, 0]
    )
    if section is None:
        return np.zeros((N, 2), dtype=np.float32)
    planar = section.to_2D()
    path2d = planar[0] if isinstance(planar, tuple) else planar
    coords = path2d.vertices  # shape (M,2): (y, z)
    ys, zs = coords[:, 1], coords[:, 0]
    z_lin = np.linspace(zs.min(), zs.max(), N)
    halfb = np.interp(z_lin, zs, ys).astype(np.float32)
    return np.stack([halfb, z_lin.astype(np.float32)], axis=1)


def sample_section_p1(mesh, xi, N=25):
    """
    Bulbous bow region extraction: transverse slicing at x=xi,
    select the largest loop (max halfbreadth) to get a single curve.
    Returns array (N,2): [half_breadth, z]
    """
    section = mesh.section(
        plane_origin=[xi, 0, 0],
        plane_normal=[1, 0, 0]
    )
    if section is None or not hasattr(section, 'discrete'):
        return np.zeros((N, 2), dtype=np.float32)
    loops = section.discrete  # list of (M_i,3) arrays
    if len(loops) == 0:
        return np.zeros((N, 2), dtype=np.float32)
    # choose loop with maximum half-breadth
    best = max(loops, key=lambda arr: np.max(np.abs(arr[:, 1])))
    coords = best  # (M,3): (x, y, z)
    ys = coords[:, 1]  # y positions (may be ±)
    zs = coords[:, 2]  # vertical positions
    z_lin = np.linspace(zs.min(), zs.max(), N)
    halfb = np.interp(z_lin, zs, np.abs(ys)).astype(np.float32)
    return np.stack([halfb, z_lin.astype(np.float32)], axis=1)


def geometry_encoding(mesh, stations, N=25, beam=1.0):
    """
    Build geometry matrices X, Y, Z ∈ R^(N×E):
      X_{ij} = station fraction x_i
      Y_{ij} = half-breadth / (beam/2)
      Z_{ij} = vertical level z
    Uses sample_section_p1 for xi<0.1, sample_section otherwise.
    """
    E = len(stations)
    X = np.zeros((N, E), dtype=np.float32)
    Y = np.zeros((N, E), dtype=np.float32)
    Z = np.zeros((N, E), dtype=np.float32)
    for j, xi in enumerate(stations):
        if xi < 0.1:
            sec = sample_section_p1(mesh, xi, N)
        else:
            sec = sample_section(mesh, xi, N)
        halfb, zvals = sec[:, 0], sec[:, 1]
        X[:, j] = xi
        Y[:, j] = halfb / (beam / 2.0)
        Z[:, j] = zvals
    return X, Y, Z


def compute_moment_invariants(X, Y, Z):
    """
    Compute central, scale-invariant 3D moments up to order 4 (p+q+r≤4).
    Returns vector length 35: MI_pqr = μ_pqr / (μ_000^(1+(p+q+r)/3)).
    """
    xs = X.flatten()
    ys = Y.flatten()
    zs = Z.flatten()
    # centralize
    xc = xs - xs.mean()
    yc = ys - ys.mean()
    zc = zs - zs.mean()
    mu000 = xs.size
    moments = []
    for p in range(5):
        for q in range(5 - p):
            for r in range(5 - p - q):
                mu = (xc**p * yc**q * zc**r).sum()
                mi = mu / (mu000 ** (1.0 + (p + q + r) / 3.0))
                moments.append(np.float32(mi))
    return np.array(moments, dtype=np.float32)


def create_sst(stl_path, N=25):
    """
    Main entry: given a hull STL, returns:
      X, Y, Z ∈ R^(N×56) geometry matrices
      M ∈ R^(35,) moment invariants
      stations ∈ R^(56,) station fractions
    """
    mesh, (L, B, D) = load_and_normalize_mesh(stl_path)
    stations = get_stations()
    X, Y, Z = geometry_encoding(mesh, stations, N, beam=B)
    M = compute_moment_invariants(X, Y, Z)
    return {
        'X': X,
        'Y': Y,
        'Z': Z,
        'M': M,
        'stations': stations
    }

# Example:
# sst = create_sst('/mnt/data/cargo vessel1.stl')
# print(sst['Y'].shape, sst['M'].shape)
