import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from marine_weather.marine_weather_queries import get_hindcast_for_vessel_track
from marine_weather.weathercube_helpers import get_all_z4_cubes_for_globe, create_global_dataset_from_z4_cubes
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file
load_dotenv()
SOFAR_API_TOKEN = os.getenv("SOFAR_API_TOKEN")
SOFAR_API_URL = "https://api.sofarocean.com"

RAW_FILE=False
VARIABLE = 'significantWaveHeight'
first_start_date = pd.Timestamp(datetime(2026, 1, 4, 6))

all_data = []
for hour in range(6):
    start_date = first_start_date + timedelta(hours=hour*6)
    end_date = start_date+timedelta(hours=1)

    date_string = start_date.strftime("%Y%m%d")
    hour_string = str(start_date.hour).zfill(2)

    if RAW_FILE:
        filename = "/Users/isabelhoughton/Documents/GitRepos/observation-processing/workdir/raw_data/CYGNSS_L2_V3.2/cyg.ddmi.s20260110-000000-e20260110-235959.l2.wind-mss.a32.d33.nc"
    else:
        filename = f"/Users/isabelhoughton/Documents/GitRepos/observation-processing/workdir/processed_data/{date_string}/{hour_string}/{VARIABLE}/CYGNSS_L2_V3.2.{date_string}.{hour_string}.{VARIABLE}.nc"

    ds = xr.open_dataset(filename)

    if RAW_FILE:
        ds = ds.rename(
                        {"lat": "latitude",
                         "lon": "longitude",
                         "swh": "significantWaveHeight",
                         "wind_speed": "windSpeed10Meter",
                         "sample_time": "time"}
                        )
    else:
        ds = ds.rename({"observation_value": VARIABLE})




    weathercube_file = f"data/weather_cube_{start_date}_hindcast_{VARIABLE}.nc"
    if os.path.exists(weathercube_file):
        global_weather = xr.open_dataset(weathercube_file)
    else:

        if VARIABLE=="windSpeed10Meter":

            all_cubes_u = get_all_z4_cubes_for_globe(init_time=start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                                         model="ECMWFHRes",
                                         variable="windVelocity10MeterEastward",
                                         token=SOFAR_API_TOKEN,
                                         resolution="high",
                                         data_type="hindcast",
                                         host=SOFAR_API_URL
                                         )
            global_weather = create_global_dataset_from_z4_cubes(all_cubes_u)

            all_cubes_v = get_all_z4_cubes_for_globe(init_time=start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                                         model="ECMWFHRes",
                                         variable="windVelocity10MeterNorthward",
                                         token=SOFAR_API_TOKEN,
                                         resolution="high",
                                         data_type="hindcast",
                                         host=SOFAR_API_URL
                                         )
            global_weather_v = create_global_dataset_from_z4_cubes(all_cubes_v)

            global_weather["windSpeed10Meter"] = np.sqrt(global_weather_v["windVelocity10MeterNorthward"]**2 + global_weather["windVelocity10MeterEastward"]**2)


            # Remove None values from attributes (netCDF doesn't support None)
            for var in global_weather.data_vars:
                global_weather[var].attrs = {k: v for k, v in global_weather[var].attrs.items() if v is not None}
            global_weather.attrs = {k: v for k, v in global_weather.attrs.items() if v is not None}
            # Convert timezone-aware times to timezone-naive for netCDF compatibility
            if global_weather['time'].dtype.name.endswith('UTC]'):
                global_weather['time'] = pd.to_datetime(global_weather['time'].values).tz_localize(None)
            global_weather.to_netcdf(weathercube_file)

        else:
            all_cubes = get_all_z4_cubes_for_globe(init_time=start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                                         model="ECMWFHRes",
                                         variable=VARIABLE,
                                         token=SOFAR_API_TOKEN,
                                         resolution="high",
                                         data_type="hindcast",
                                         host=SOFAR_API_URL
                                         )
            global_weather = create_global_dataset_from_z4_cubes(all_cubes)
            # Remove None values from attributes (netCDF doesn't support None)
            for var in global_weather.data_vars:
                global_weather[var].attrs = {k: v for k, v in global_weather[var].attrs.items() if v is not None}
            global_weather.attrs = {k: v for k, v in global_weather.attrs.items() if v is not None}
            # Convert timezone-aware times to timezone-naive for netCDF compatibility
            if global_weather['time'].dtype.name.endswith('UTC]'):
                global_weather['time'] = pd.to_datetime(global_weather['time'].values).tz_localize(None)
            global_weather.to_netcdf(weathercube_file)



    ds_filtered = ds # ds.where((pd.to_datetime(ds.time.values[0]) >= start_date) & (pd.to_datetime(ds.time.values[0]) <= end_date), drop=True)

    cygnss_df = ds_filtered[["latitude", "longitude", "time", VARIABLE]].to_dataframe().reset_index()



    # fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    # ax.coastlines()
    # ax.add_feature(cfeature.LAND, alpha=0.3)
    # ax.add_feature(cfeature.OCEAN, alpha=0.3)
    #
    # # Plot your data
    # scatter = ax.scatter(
    #     cygnss_df["longitude"],
    #     cygnss_df["latitude"],
    #     c=cygnss_df[VARIABLE],
    #     cmap="Spectral_r",
    #     s=50,
    #     transform=ccrs.PlateCarree(),
    # )
    # plt.colorbar(scatter, ax=ax, label="Hs")
    # plt.title(f"CYGNSS Data on {start_date}")
    # plt.show()


    obs_lons = xr.DataArray(cygnss_df['longitude'].values, dims='obs')
    obs_lons[obs_lons>180] -= 360
    obs_lats = xr.DataArray(cygnss_df['latitude'].values, dims='obs')
    # Convert to timezone-naive datetime64[ns] to avoid TypeError
    obs_times_raw = pd.to_datetime(cygnss_df['time'].values)

    if obs_times_raw.tz is not None:
        obs_times_raw = obs_times_raw.tz_convert(None)
    obs_times = xr.DataArray(obs_times_raw, dims='obs')


    model_swh = global_weather.interp(
        latitude=obs_lats,
        longitude=obs_lons,
        time=obs_times,
        method='linear'
    )[VARIABLE].values

    # Add model values to dataframe
    cygnss_df[f'model_{VARIABLE}'] = model_swh

    all_data.append(cygnss_df)
    print(f"\nExtracted {len(cygnss_df)} model values")


final_df = pd.concat(all_data, ignore_index=True)

plt.scatter(final_df[VARIABLE], final_df[f"model_{VARIABLE}"], alpha=0.1)
plt.plot([0,max(final_df[VARIABLE])], [0,max(final_df[VARIABLE])], '--', alpha=0.5, zorder=10, c='k')
plt.xlabel(f'CYGNSS {VARIABLE}')
plt.ylabel(f"ECMWFHRes {VARIABLE}")
plt.show()
print('Done!')
