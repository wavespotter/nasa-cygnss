import os
import os.path
import pandas as pd
from datetime import datetime, timedelta
from roguewave.spotterapi import get_spotter_data, get_spotter_ids
import xarray as xr
from pandas import DataFrame

def extract_cygnss(start_date: datetime,
                   end_date: datetime,
                   data_file_source: str = "/Users/isabelhoughton/Documents/GitRepos/observation-processing/workdir/processed_data/",
                   variables = None,
                   satellite_name = "CYGNSS_L2_V3.2",
                   recalculate = False
                   )-> DataFrame:

    save_file_path = f'data/{satellite_name}_{start_date.strftime("%Y%m%d%H")}_{end_date.strftime("%Y%m%d%H")}.csv'

    if os.path.exists(save_file_path) and not recalculate:
        all_cygnss_df = pd.read_csv(save_file_path, parse_dates=["time"])
        print(f'Loading {satellite_name} from {save_file_path}')
    else:
        print(f'Re-processing {satellite_name}')
        if variables is None:
            variables = [ "meanSquareSlope", "meanSquareSlopeUncertainty", "windSpeed10Meter",] #  "significantWaveHeight"]

        all_cygnss_df = []
        for hourly_date in pd.date_range(start_date, end_date, freq="h"):
            date_string = hourly_date.strftime("%Y%m%d")
            hour_string = str(hourly_date.hour).zfill(2)
            filepath = os.path.join(data_file_source, date_string, hour_string)
            all_df = []
            for var in variables:
                try:
                    ds = xr.open_dataset(os.path.join(filepath, var, f"{satellite_name}.{date_string}.{hour_string}.{var}.nc"))
                    ds = ds.rename({'observation_value': f'{satellite_name}_{var}'})
                    ds = ds.drop_vars(["observation_error", "observation_type", "platform"])

                    df = ds.to_dataframe().reset_index()
                    df = df.drop(columns=['observation_number'], errors='ignore')
                    df = df.drop_duplicates(subset=['latitude', 'longitude', 'time'], keep='first')


                    all_df.append(df)
                except FileNotFoundError as e:
                    print(e)

            if len(all_df) < 1:
                continue
            else:
                merged_df = all_df[0]
                for df in all_df[1:]:
                    merged_df = merged_df.merge(df, on=['latitude', 'longitude', 'time'], how='inner')


                # merged_df = xr.merge(var_dict, join='inner', compat='override').to_dataframe()
                # merged_df["time"] = hourly_date
                all_cygnss_df.append(merged_df)

        all_cygnss_df = pd.concat(all_cygnss_df, ignore_index=True)
        all_cygnss_df["longitude"] = (all_cygnss_df["longitude"] + 180) % 360 - 180
        all_cygnss_df.to_csv(save_file_path, index=False)

    return all_cygnss_df