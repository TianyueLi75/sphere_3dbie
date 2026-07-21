"""Remake the container-obstacle timing/self-convergence plot from the data recorded
at the end of Stk3d_container_obstacle.py, adding an O(lmax^3) reference fit line
(2.47e-6 * lmax^3) to the coupled-solve timing curve.

Run (from repo root):
    python timing/Stk3d_container_obstacle_fitted.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

# Data recorded at the end of Stk3d_container_obstacle.py.
lmax_list = np.array([4, 8, 16, 32, 64, 128, 256, 512])
t_periter_solve = np.array([
    0.0111619234085083, 0.024657130241394043, 0.05894827842712402,
    0.14652353525161743, 0.5968244075775146, 3.9781848192214966,
    32.544267535209656, 331.89166649182636,
])
t_eval = np.array([3.113, 2.175, 2.267, 2.407, 2.582, 3.306, 4.695, 14.345])
# Self-convergence max rel diff vs lmax=512 (reference is 0 by construction).
err = np.array([3.355e-02, 4.532e-04, 1.381e-08, 3.369e-15,
                4.115e-15, 2.297e-15, 2.144e-15, 0.0])


def plot_timing_convergence(lmax_list, t_solve, t_eval, err, title, path):
    """Combined loglog figure: timing on the left axis, self-convergence relative
    difference on the right axis, sharing a log lmax axis. The reference point
    (finest lmax, zero diff by construction) is dropped from the error curve.
    An O(lmax^3) fit line is overlaid on the coupled-solve timing."""
    fig, ax1 = plt.subplots()
    l1 = ax1.loglog(lmax_list, t_solve, 'k+-', label="coupled solve")
    l2 = ax1.loglog(lmax_list, t_eval, 'ko-', label="off-surf eval")
    l4 = ax1.loglog(lmax_list, 2.47e-6 * lmax_list**3, 'k--',
                    label=r"$\mathcal{O}(l_{\text{max}}^3)$")
    ax1.set_xlabel("lmax"); ax1.set_ylabel("Time (s)")

    ax2 = ax1.twinx()
    l3 = ax2.loglog(lmax_list[:-1], err[:-1], 'r*--',
                    label=f"max rel diff vs lmax={lmax_list[-1]}")
    ax2.set_ylabel(f"max rel diff vs lmax={lmax_list[-1]}")

    lines = l1 + l2 + l4 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines])
    ax1.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    plot_timing_convergence(
        lmax_list, t_periter_solve, t_eval, err,
        "Container-obstacle Stokes: timing + self-convergence",
        os.path.join(here, "./plots/Stk3d_container_obstacle_fitted.svg"))
    print("Wrote Stk3d_container_obstacle_fitted.svg to", os.path.join(here, "plots"))
