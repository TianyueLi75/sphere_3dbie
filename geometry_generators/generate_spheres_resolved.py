"""Generate non-intersecting polydisperse spheres inside a spherical container.

Python port of generateSpheresResolved.m / spheres.m (without plotting). Spheres are
confined inside a ball of radius ``outer_radius`` centered at the origin, kept clear of
each other and of the container wall by at least ``min_separation_distance``.

Run as a script to reproduce the demo (~100 spheres in a unit ball) and write the
result as a 4-column ``[x y z r]`` text file into this folder, byte-compatible with the
MATLAB output of spheres.m::

    python geometry_generators/generate_spheres_resolved.py
"""

import os

import numpy as np

MAX_ATTEMPTS_PER_SPHERE = 5000  # Max attempts to place a single sphere


def generate_spheres_resolved(num_target_spheres, min_r, max_r, outer_radius,
                              min_separation_distance, rng=None):
    """Place non-intersecting spheres inside a spherical container.

    Parameters
    ----------
    num_target_spheres : int
        Desired number of interior spheres to generate.
    min_r, max_r : float
        Minimum / maximum radius of an interior sphere.
    outer_radius : float
        Radius of the spherical container (centered at the origin).
    min_separation_distance : float
        Minimum gap between sphere surfaces (and between a sphere surface and the wall).
    rng : numpy.random.Generator, optional
        Random generator (for reproducibility). Defaults to ``np.random.default_rng()``.

    Returns
    -------
    numpy.ndarray
        ``(N, 4)`` array; each row is ``[x, y, z, r]`` for a placed sphere.
    """
    if rng is None:
        rng = np.random.default_rng()

    spheres = []  # list of [x, y, z, r]

    print(f"Attempting to generate {num_target_spheres} non-intersecting spheres "
          f"in a sphere of radius {outer_radius:.4f}...")

    for i in range(num_target_spheres):
        found_position = False
        attempts = 0

        while not found_position and attempts < MAX_ATTEMPTS_PER_SPHERE:
            attempts += 1

            # 1. Random radius
            r_new = min_r + (max_r - min_r) * rng.random()

            # 2. Random center uniformly inside the ball of radius
            #    (outer_radius - r_new - min_separation_distance) so the new sphere
            #    stays fully inside the container, clear of the wall.
            R_place = outer_radius - r_new - min_separation_distance
            if R_place <= 0:
                continue  # sphere too big to fit in the container

            # Uniform-in-ball: random direction * radius scaled by rand^(1/3)
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            radial = R_place * rng.random() ** (1.0 / 3.0)
            center_new = radial * direction

            # 3. Check for intersection with previously placed spheres (Euclidean)
            intersects = False
            for x, y, z, r_existing in spheres:
                dist = np.linalg.norm(center_new - np.array([x, y, z]))
                if dist < r_new + r_existing + min_separation_distance:
                    intersects = True
                    break

            # 4. Accept if clear
            if not intersects:
                spheres.append([center_new[0], center_new[1], center_new[2], r_new])
                found_position = True

        if not found_position:
            print(f"Warning: Could not place sphere {i + 1} after "
                  f"{MAX_ATTEMPTS_PER_SPHERE} attempts.")
            print(f"         Generation stopped at {len(spheres)} spheres. "
                  f"This usually means the packing limit is reached.")
            break

    print(f"Successfully generated {len(spheres)} non-intersecting spheres.")
    return np.array(spheres, dtype=float).reshape(-1, 4)


if __name__ == "__main__":
    # --- Container + packing parameters (mirrors spheres.m) ---
    outer_radius = 1.0              # radius of the spherical container
    Nptcl = 10                     # number of interior spheres to attempt
    min_r = 0.1                    # minimum interior-sphere radius (~0.1)
    max_r = 0.2                    # maximum interior-sphere radius (~0.1)
    min_separation_distance = 0.001  # minimum surface-to-surface gap

    rng = np.random.default_rng(1)  # seeded for reproducibility

    sphere_data = generate_spheres_resolved(
        Nptcl, min_r, max_r, outer_radius, min_separation_distance, rng=rng)

    # --- Report total volume + volume fraction ---
    tot_vol = float(np.sum(4.0 * np.pi * sphere_data[:, 3] ** 3 / 3.0))
    container_vol = 4.0 * np.pi * outer_radius ** 3 / 3.0
    print(f"\n Total interior-sphere volume is {tot_vol:f}.")
    print(f" Volume fraction (relative to container) is {tot_vol / container_vol:f}.")

    # --- Write to file in this folder, matching the MATLAB %.8f format ---
    here = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(here, f"sphere_data_{sphere_data.shape[0]}_ball.txt")
    np.savetxt(filename, sphere_data, fmt="%.8f",
               header=f"outer_radius = {outer_radius:.8f}")
    print(f" Wrote {sphere_data.shape[0]} spheres to {filename}")
