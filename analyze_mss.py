import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np


cygnss_df = pd.read_csv("data/cygnss_20251212_12.csv")
spotter_df = pd.read_csv("data/spotter_20251212_12.csv")

cygnss_tree = cKDTree(cygnss_df[["latitude", "longitude"]].values)
cygnss_df["matching_spotter_ind"] = np.nan

# for the cygnss df, have a column that points to the spotter df index that is a co-located spotter
for spotter_ind, row in spotter_df.iterrows():
    target_point = np.array([row.latitude, row.longitude])

    # using euclidean distance will allow further colocation at the poles
    # cygnss is low latitudes (+/-40) so shouldn't be critical
    indices = cygnss_tree.query_ball_point(target_point, r=0.5)
    cygnss_df.loc[indices, "matching_spotter_ind"] = spotter_ind


colocated_df = cygnss_df.loc[cygnss_df.matching_spotter_ind.notna()]
print(len(colocated_df), " total matches found.")


colocated_df["spotter_mss"] = spotter_df["meanSquareSlope"].take(list(colocated_df.matching_spotter_ind.values.astype(int))).values
colocated_df["spotter_lat"] = spotter_df["latitude"].take(list(colocated_df.matching_spotter_ind.values.astype(int))).values
colocated_df["spotter_lon"] = spotter_df["longitude"].take(list(colocated_df.matching_spotter_ind.values.astype(int))).values
colocated_df["spotter_swh"] = spotter_df["significantWaveHeight"].take(list(colocated_df.matching_spotter_ind.values.astype(int))).values



colocated_df.plot.scatter(
    x="observation_value_meanSquareSlope",
    y="spotter_mss",
    marker='.',
)
plt.show()



fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(colocated_df['spotter_mss'], colocated_df['observation_value_meanSquareSlope'],
            yerr=colocated_df['observation_value_meanSquareSlopeUncertainty'],
            fmt='o',
            capsize=3,
            linestyle='none',  # no line connecting points
            alpha=0.7)
scat = ax.scatter(colocated_df['spotter_mss'], colocated_df['observation_value_meanSquareSlope'],
            c=colocated_df['observation_value_windSpeed10Meter'],
            cmap="Spectral_r",
            alpha=1, zorder=10)
plt.colorbar(scat, label="Significant Wave Height")
ax.set_xlabel('Spotter')
ax.set_ylabel('CYGNSS')
ax.grid(True, alpha=0.3)
plt.show()



import numpy as np
# cygnss_df = cygnss_df.loc[cygnss_df.observation_value_meanSquareSlopeUncertainty < 0.01]
x, y = cygnss_df['observation_value_windSpeed10Meter'], cygnss_df['observation_value_meanSquareSlope']
coeffs = np.polyfit(x, y, deg=1)
slope, intercept = coeffs[0], coeffs[1]

fn = np.poly1d(coeffs)
y_pred = fn(x)

plt.scatter(cygnss_df['observation_value_windSpeed10Meter'], cygnss_df['observation_value_meanSquareSlope'])
plt.scatter(x, y_pred)
# plt.ylim(0, 0.02)
plt.show()

print('done')
