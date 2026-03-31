
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.colors import BoundaryNorm
import matplotlib.cm as cm


def plot_x_vs_y(df, x_col, y_col, c_col=None):
    fig,ax = plt.subplots(1,1, figsize=(5,4))
    if c_col is not None:
        label=c_col
        c_col = df[c_col]

        # Create discrete bins of width 3
        c_binned = (c_col / 3).round(0) * 3

        # Create boundaries for discrete colormap
        vmin = np.floor(c_binned.min() / 3) * 3
        vmax = np.ceil(c_binned.max() / 3) * 3
        bounds = np.arange(vmin, vmax + 3, 3)

        # Use only the richer part of inferno (0 to 0.85) to avoid pale yellow
        inferno_truncated = cm.get_cmap('inferno', 256)(np.linspace(0, 0.85, 256))
        cmap_truncated = cm.colors.ListedColormap(inferno_truncated)
        norm = BoundaryNorm(bounds, cmap_truncated.N)

        pcm = ax.scatter(df[x_col], df[y_col], c=c_binned, cmap=cmap_truncated, norm=norm, s=5)
        plt.colorbar(pcm, ax=ax, label=label, boundaries=bounds, ticks=bounds)
    else:
        pcm = ax.scatter(df[x_col], df[y_col], s=5)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    plt.savefig(f"plots/{x_col}_vs_{y_col}.png", dpi=250)
    plt.show()

def binned_plot(df, x_col, y_col, show=True):
    bin_means, bin_edges, _ = binned_statistic(df[x_col], df[y_col], statistic='mean', bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot
    plt.scatter(df[x_col], df[y_col], alpha=0.3, label='Data', marker='.')
    plt.plot(bin_centers, bin_means, c='k', marker='.', linewidth=2, label='Binned average')
    plt.legend()
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if show:
        plt.show()



def colocated_mss_comparions(final_df, satellite_name ="CYGNSS_L2_V3.2", title_note=''):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Calculate point density for coloring
    x = final_df[f"{satellite_name}_meanSquareSlope"].values
    y = final_df[f"meanSquareSlope"].values
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    pcm = ax.scatter(x, y, c=z, s=5, cmap='bone_r', vmin=-1000, vmax=z.max())
    # cbar = plt.colorbar(pcm, ax=ax, label="Density", extend='min')  # extend='min' adds arrow at bottom
    ax.set_xlabel(f"{satellite_name} mean square slope []")
    ax.set_ylabel("Spotter mean square slope []")

    # Fit a line to the data, weighted by density
    coeffs = np.polyfit(x, y, 1, w=z)  # Linear fit weighted by density: y = mx + b
    slope, intercept = coeffs
    fit_line = np.poly1d(coeffs)

    # Plot the fit line
    x_fit = np.linspace(x.min(), x.max(), 100)
    yfit = fit_line(x_fit)
    ax.plot(x_fit, yfit, 'lightblue', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}', alpha=0.7, ls='--')

    minimum_max = np.min([np.max(x), np.max(y)])
    ax.plot([np.min(x), np.max(yfit)], [np.min(x), np.max(yfit)], c='k', alpha=0.5, ls='--', label='1:1 line')

    ax.legend()
    plt.title(f"Co-located mean square slope comparison\n{title_note}")
    plt.tight_layout()
    plt.savefig("plots/scatter_comparison_mss", dpi=250)
    plt.show()

def plot_mss_vs_windspeed(final_df, satellite_name="CYGNSS_L2_V3.2", title_note=''):
    x_col = f"{satellite_name}_windSpeed10Meter"
    y_col_cyg = f"{satellite_name}_meanSquareSlope"
    y_col_spot = f"meanSquareSlopeExtended"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left subplot: MSS vs Wind Speed for both datasets
    for y_col, name, cc, cc2 in zip([y_col_spot, y_col_cyg], ["Spotter", satellite_name],
                                    ["darkslategrey", "darkgoldenrod"],
                                    ["cadetblue", "goldenrod"]):
        bin_means, bin_edges, _ = binned_statistic(final_df[x_col], final_df[y_col], statistic='mean', bins=20)
        bin_stds, _, _ = binned_statistic(final_df[x_col], final_df[y_col], statistic='std', bins=20)
        bin_counts, _, _ = binned_statistic(final_df[x_col], final_df[y_col], statistic='count', bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Filter bins with at least 10 observations
        valid_bins = bin_counts >= 20
        bin_centers_filtered = bin_centers[valid_bins]
        bin_means_filtered = bin_means[valid_bins]
        bin_stds_filtered = bin_stds[valid_bins]

        # Plot
        ax1.scatter(final_df[x_col], final_df[y_col], alpha=0.3, marker='.', s=2, c=cc2)
        eb = ax1.errorbar(bin_centers_filtered, bin_means_filtered, yerr=bin_stds_filtered, marker='.', linewidth=2,
                          alpha=0.8, c=cc, capsize=3, label=name, elinewidth=1.5)
        # Make error bars transparent
        eb[-1][0].set_alpha(0.5)  # Error bar lines
        if len(eb) > 2:
            for cap in eb[1]:  # Caps
                cap.set_alpha(0.5)

    ax1.legend()
    ax1.set_xlabel("Wind Speed (CYGNSS) [m/s]")
    ax1.set_ylabel("mean square slope [-]")

    # Right subplot: Ratio of CYGNSS/Spotter vs Wind Speed
    final_df['mss_ratio'] = final_df[y_col_cyg] / final_df[y_col_spot]

    # Calculate binned statistics for ratio
    bin_ratio_means, bin_edges, _ = binned_statistic(final_df[x_col], final_df['mss_ratio'], statistic='mean', bins=20)
    bin_ratio_stds, _, _ = binned_statistic(final_df[x_col], final_df['mss_ratio'], statistic='std', bins=20)
    bin_ratio_counts, _, _ = binned_statistic(final_df[x_col], final_df['mss_ratio'], statistic='count', bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter bins with at least 10 observations
    valid_bins = bin_ratio_counts >= 20
    bin_centers_filtered = bin_centers[valid_bins]
    bin_ratio_means_filtered = bin_ratio_means[valid_bins]
    bin_ratio_stds_filtered = bin_ratio_stds[valid_bins]

    # Plot ratio
    ax2.scatter(final_df[x_col], final_df['mss_ratio'], alpha=0.3, marker='.', s=2, c='darkgray')
    eb2 = ax2.errorbar(bin_centers_filtered, bin_ratio_means_filtered, yerr=bin_ratio_stds_filtered, marker='.',
                       linewidth=2, alpha=0.8, c='k', capsize=3, elinewidth=1.5)
    # Make error bars transparent
    eb2[-1][0].set_alpha(0.5)
    if len(eb2) > 2:
        for cap in eb2[1]:
            cap.set_alpha(0.5)

    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='1:1 ratio')

    # Add mean and median lines
    ratio_mean = final_df['mss_ratio'].mean()
    ratio_median = final_df['mss_ratio'].median()
    ax2.axhline(y=ratio_median, color='darkblue', linestyle='--', alpha=0.4, label=f'Median: {ratio_median:.2f}')

    ax2.legend()
    ax2.set_xlabel("Wind Speed (CYGNSS) [m/s]")
    ax2.set_ylabel("CYGNSS mss / Spotter mss [-]")

    plt.suptitle(f"mss versus windspeed\n{title_note}")

    plt.tight_layout()
    plt.savefig("plots/mss_vs_windspeed_and_ratio.png", dpi=250)
    plt.show()



