# Takes the output from compute_ripples_CSD.py and renders subplots

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def Plot_CSD_results(csd_results, session_title="CSD Results"):
    """Direct translation of Plot_CSD_results.m.

    Plots the mean CSD vertical profile (Subplot 1) and the 2D spatiotemporal
    CSD heatmap (Subplot 2).
    """
    csd_matrix = csd_results["csd_matrix"]
    mean_csd_per_channel = csd_results["mean_csd_per_channel"]
    time_vector = csd_results["time_vector"]
    num_channels = csd_results["num_channels"]

    channels = np.arange(1, num_channels + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(session_title, fontsize=14, fontweight="bold")

    # --- SUBPLOT 1: Vertical Source/Sink Profile ---
    # Create smooth spline interpolation across channel depth
    y_channels = np.linspace(1, num_channels, len(mean_csd_per_channel))
    y_dense = np.linspace(1, num_channels, 200)
    interp_func = interp1d(y_channels, mean_csd_per_channel, kind="cubic")
    csd_dense = interp_func(y_dense)

    # Plot interpolated line and discrete channel markers
    axs[0].plot(csd_dense, y_dense, color="black", linewidth=1.5)
    axs[0].scatter(
        mean_csd_per_channel,
        y_channels,
        color="black",
        s=20,
        zorder=3,
        label="Channels",
    )

    # Fill sources (>0) red and sinks (<0) blue
    axs[0].fill_betweenx(
        y_dense,
        0,
        csd_dense,
        where=(csd_dense >= 0),
        color="red",
        alpha=0.35,
        label="Source (>0)",
    )
    axs[0].fill_betweenx(
        y_dense,
        0,
        csd_dense,
        where=(csd_dense < 0),
        color="blue",
        alpha=0.35,
        label="Sink (<0)",
    )

    axs[0].axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
    axs[0].invert_yaxis()  # Channel 1 at top (surface)
    axs[0].set_xlabel("Mean CSD (Sources > 0 | Sinks < 0)")
    axs[0].set_ylabel("Channel / Depth")
    axs[0].set_title(f"Mean CSD Profile ({num_channels} Channels)")
    axs[0].grid(True, linestyle=":", alpha=0.6)
    axs[0].legend(loc="lower right", fontsize=8)

    # --- SUBPLOT 2: 2D Spatiotemporal CSD Heatmap ---
    vmax = np.max(np.abs(csd_matrix))  # Symmetric color limits centered at zero

    im = axs[1].imshow(
        csd_matrix,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        extent=[time_vector[0], time_vector[-1], num_channels, 1],
    )

    cbar = fig.colorbar(im, ax=axs[1])
    cbar.set_label("CSD Amplitude")

    axs[1].axvline(
        0, color="black", linestyle="--", alpha=0.7, label="Ripple Peak"
    )
    axs[1].set_xlabel("Time relative to peak (ms)")
    axs[1].set_ylabel("Channel")
    axs[1].set_title("Spatiotemporal CSD Map")
    axs[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    # --- Save line commented out for testing ---
    # plt.savefig("CSD_results.png", dpi=300, bbox_inches='tight')

    plt.show()