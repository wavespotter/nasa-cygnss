# NASA CYGNSS Mean Square Slope Analysis

Scripts for comparing CYGNSS satellite mean square slope (MSS) observations against Spotter buoy measurements.

---

## Scripts

### `driver_mss.py`
**Main entry point.** Orchestrates the full pipeline. You must first run `ingest_observations.py` in the repo `observation-processing` for the date ranges you are interested in (set in constants.py) in order to have the raw files from PODAAC available locally (or download them yourself)
1. Fetches Spotter buoy MSS data for a given date range
2. Fetches and loads CYGNSS L2 data via `save_cygnss.py`
3. Co-locates the two datasets in space (20 km radius) and time (±1 hour)
4. Applies QA/QC filtering (removes observations where MSS < N × MSS uncertainty)
5. Saves the combined co-located dataset to `data/`
6. Produces comparison plots via `plotting.py`

Set `REQUERY = False` to load a previously saved combined CSV instead of re-running colocation.

---

### `save_cygnss.py`
Loads and preprocesses CYGNSS L2 NetCDF files from a local processed data directory. Iterates hourly over a date range, reads per-variable NetCDF files (`meanSquareSlope`, `meanSquareSlopeUncertainty`, `windSpeed10Meter`), merges them into a single DataFrame, and saves to CSV. Normalises longitudes to [−180, 180].

---

### `plotting.py`
Reusable plotting utilities:
- **`plot_x_vs_y`** – Scatter plot of any two columns, with optional discrete colour-coded third variable (wind speed).
- **`binned_plot`** – Scatter with overlaid binned mean.
- **`colocated_mss_comparions`** – Density-coloured scatter of CYGNSS MSS vs Spotter MSS, with a weighted linear fit and 1:1 reference line.
- **`plot_mss_vs_windspeed`** – Two-panel plot: (left) MSS vs wind speed for both instruments with binned error bars; (right) CYGNSS/Spotter MSS ratio vs wind speed.

---

### `eda_mss_disagreement.py`
Exploratory analysis of what drives disagreement between CYGNSS and Spotter MSS/wind-speed estimates. Loads the combined co-located CSV, fits a curve to the CYGNSS–Spotter MSS relationship (power-law or tanh), and computes residuals. Produces:
- Correlation heatmap of residuals vs. latitude, longitude, drift speed, SWH, swell height, and MSS uncertainty
- Scatter grids of each variable vs. MSS residual and wind-speed difference
- Spatial map of residuals coloured by lat/lon
- Time series of MSS residual and wind-speed difference

Switch between fit methods with `FIT_METHOD = "power"` or `"tanh"`.

---

### `extend_spectra.py`
Exploratory script investigating the effect of spectral tail extension on MSS. Fetches a single Spotter frequency spectrum, extends the tail to 1.64 Hz using a `f^-5` power law, and compares the resulting MSS against the original truncated-spectrum value. Includes log-log plots of the raw and extended spectra.


---

## Data (`data/`)

| File pattern | Description |
|---|---|
| `CYGNSS_L2_V3.2_*.csv` | Processed CYGNSS observations for a date range |
| `combined_spotter_CYGNSS_L2_V3.2_*.csv` | Co-located CYGNSS + Spotter observations |
| `spotter_mss_*.csv` | Spotter MSS for a date range |
| `weather_cube_*_hindcast*.nc` | Cached ECMWFHRes hindcast weather cubes |

## Plots (`plots/`)

Generated figures are saved to `plots/` (MSS scatter, MSS vs wind speed, correlation matrices, pair plots) and `plots/eda/` (EDA diagnostic plots from `eda_mss_disagreement.py`).

