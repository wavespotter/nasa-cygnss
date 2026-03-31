import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import xarray as xr
from dask.array import frexp
from roguewave.spotterapi import get_spotter_data, get_spotter_ids
from roguewave import Spectrum
# from roguewavespectrum.spotter import sp
from roguewavespectrum.spotter._spotter_extrapolate_tail import extrapolate_tail
spot_id = "SPOT-1112"
start_date = datetime(year=2025, month=12, day=12, hour=6)
end_date = datetime(year=2025, month=12, day=13, hour=6)

all_spectra = get_spotter_data(
    spotter_ids=[spot_id,],
    start_date=start_date,
    end_date=end_date,
    data_type="frequencyData"
)


spec = all_spectra[spot_id].isel(time=0)

def calc_mss(frequency, variance_density, g=9.81, pi=3.14):
    return np.trapezoid(variance_density*frequency**4, frequency)*(2*np.pi)**4 / g**2

freq = spec.frequency.values
ef = spec.variance_density.values
print(spec.mean_squared_slope())


extended_freq = np.linspace(freq[-1]+0.01, 1.64, 20)
extended_ef_compensated = np.ones(len(extended_freq))*(ef*freq**5)[-1]
extended_ef = extended_ef_compensated/(extended_freq**5)

full_freq = np.concat([freq, extended_freq])
full_ef = np.concat([ef, extended_ef])
print('done')

plt.loglog(full_freq, full_ef*full_freq**5, marker='.')
plt.loglog(freq, ef*freq**5, marker='.')
plt.xlabel('Frequency [hz]')
plt.ylabel('Variance Density [m2/s]')
plt.show()

plt.loglog(full_freq, full_ef, marker='.')
plt.loglog(freq, ef, marker='.')
plt.xlabel('Frequency [hz]')
plt.ylabel('Variance Density [m2/s]')
plt.show()

mss_original = calc_mss(frequency=freq, variance_density=ef)
mss_extended = calc_mss(frequency=full_freq, variance_density=full_ef)

print(f"auto mss: {spec.mean_squared_slope()}\n"
      f"original mss: {mss_original}\n"
      f"extended mss: {mss_extended}")