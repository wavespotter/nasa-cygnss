"""
this is a main script to ultimately get a dataframe of colocated observations

- get spotter spectra in a dataframe for time range
- get cygnss data in a dataframe for a time range
- colocate

"""


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from roguewave.colocate.pointdata import colocate_points

from spotter.get_spotter_spectra import get_spotter_mss_df
from save_cygnss import extract_cygnss
import os
import os.path
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr
from plotting import plot_x_vs_y, binned_plot, plot_mss_vs_windspeed, colocated_mss_comparions
from marine_weather.marine_weather_queries import get_hindcast_for_vessel_track
from scipy.stats import binned_statistic


start_date = datetime(year=2025, month=12, day=12, hour=6)
end_date = datetime(year=2026, month=1, day=29, hour=6)
satellite_name = "CYGNSS_L2_V3.2"

save_file_path = f'data/combined_spotter_{satellite_name}_{start_date.strftime("%Y%m%d%H")}_{end_date.strftime("%Y%m%d%H")}.csv'
REQUERY = False

def combined_qaqc(input_df:DataFrame,
                  uncertainty_multiplier:float=2,
                  minimum_swh: float = 0.5,
                  minimum_windspeed:float = 1,
                  satellite_name:str="CYGNSS_L2_V3.2") -> DataFrame:

    cleaned_df = input_df[input_df[f"{satellite_name}_meanSquareSlope"] > uncertainty_multiplier*input_df[f"{satellite_name}_meanSquareSlopeUncertainty"]]
    # cleaned_df = cleaned_df[cleaned_df["significantWaveHeight"] > minimum_swh]
    # cleaned_df = cleaned_df[cleaned_df["windSpeed10Meter"] > minimum_windspeed]

    # cleaned_df = cleaned_df[np.abs(cleaned_df["significantWaveHeight"]-cleaned_df[f"{satellite_name}_significantWaveHeight"] ) <0.2]
    # cleaned_df = cleaned_df[np.abs(cleaned_df["windSpeed10Meter"]-cleaned_df[f"{satellite_name}_windSpeed10Meter"] ) <1]


    return cleaned_df

if os.path.exists(save_file_path) and not REQUERY:
    print(f"Loading combined data from {save_file_path}")
    combined_df = pd.read_csv(save_file_path, parse_dates=["time", "time_spotter"])
else:
    cygnss_df = extract_cygnss(start_date=start_date,
                               end_date=end_date,
                               recalculate=False
                               )

    spotter_df = get_spotter_mss_df(start_date=start_date,
                                    end_date=end_date,
                                    recalculate=False)


    colocated_spotter_df = colocate_points(data=cygnss_df,
                                   data_to_colocate=spotter_df,
                                   radius=20*1000, # meters
                                   time_delta=timedelta(hours=1))

    combined_df = colocated_spotter_df.join(cygnss_df, how="left", rsuffix="_spotter")

    # colocated_spotter_df = colocate_points(data=spotter_df,
    #                                data_to_colocate=cygnss_df,
    #                                radius=20*1000, # meters
    #                                time_delta=timedelta(hours=1))
    #
    # combined_df = colocated_spotter_df.join(spotter_df, how="left", rsuffix="_spotter")

    combined_df = combined_df.dropna()
    combined_df.to_csv(save_file_path, index=False)


final_df = combined_qaqc(combined_df, uncertainty_multiplier=4)

# hindcast_val_v = get_hindcast_for_vessel_track(final_df, model_id="ECMWFHRes",
#                                              variable="windVelocity10MeterNorthward")
# hindcast_val_u = get_hindcast_for_vessel_track(final_df, model_id="ECMWFHRes",
#                                              variable="windVelocity10MeterEastward")
# hindcast_ws = np.sqrt(np.asarray(hindcast_val_u)**2 + np.asarray(hindcast_val_v)**2)



#
# ## PAIR PLOT ##########################################################################################################
# variables_of_interest = [col for col in final_df.columns if not any(keyword in col.lower() for
#                                                                        keyword in ['latitude', 'longitude', 'time', 'uncertainty'])]
#
# sns.pairplot(final_df, vars=variables_of_interest)
# plt.savefig('plots/pairplot.png', dpi=250)
# plt.show()
#
#
# ## CORRELATION HEATMAP ##########################################################################################################
# corr = final_df[variables_of_interest].corr()
# # Create mask for upper triangle to show only lower triangle (more concise)
# mask = np.tril(np.ones_like(corr, dtype=bool))
#
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, mask=~mask, annot=True, cmap='coolwarm', center=0,
#             fmt='.2f', square=True, linewidths=0.5)
# plt.title(f'Correlation Matrix')
# plt.tight_layout()
# plt.savefig('plots/correlation_matrix.png', dpi=250)
# plt.show()
#



plot_x_vs_y(final_df, f"{satellite_name}_meanSquareSlope", f"meanSquareSlopeExtended", f"{satellite_name}_windSpeed10Meter")



plot_mss_vs_windspeed(final_df,) #  title_note="Filtered for windspeed agreement")
# colocated_mss_comparions(final_df, )  # title_note="Filtered for windspeed agreement")

# plot_x_vs_y(final_df, f"{satellite_name}_windSpeed10Meter", f"{satellite_name}_meanSquareSlope", )
# plot_x_vs_y(final_df, f"{satellite_name}_windSpeed10Meter", f"meanSquareSlope", )

print('done')
# pcm = ax.scatter(final_df[f"{satellite_name}_meanSquareSlope"], final_df[f"meanSquareSlope"], c=final_df[f"{satellite_name}_windSpeed10Meter"])
# plt.colorbar(pcm, ax=ax, label="Wind Speed (CYGNSS)")
# ax.set_xlabel(f"{satellite_name} mss")
# ax.set_ylabel("Spotter mss")
# plt.show()



