import streamlit as st

import requests
import pandas as pd
import numpy as np
import datetime
from pymetdecoder import synop as s
from io import StringIO
import matplotlib.pyplot as plt

import xarray as xr
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
from io import BytesIO
from netCDF4 import Dataset
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


st.title("Plot Northerly Cold Surge")
if st.button("Tampilkan Grafik"):
    df_stations = pd.read_csv('https://raw.githubusercontent.com/alfuadi/SurgeIndex/refs/heads/main/SiberianStations.txt')
    backwardsteps = 30

    def quick_synop_test(begin, end, area):
        """Testing cepat pengambilan data SYNOP"""

        # URL dengan parameter spesifik
        url = "https://www.ogimet.com/cgi-bin/getsynop"
        params = {
            'begin': begin,
            'end': end,
            'state': area
        }
        response = requests.get(url, params=params, timeout=10)

        # Parse ke DataFrame
        lines = response.text.strip().split('\n')
        data = []

        for line in lines:
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 7:
                    data.append({
                        'station': parts[0],
                        'year': parts[1],
                        'month': parts[2],
                        'day': parts[3],
                        'hour': parts[4],
                        'minute': parts[5],
                        'synop': ','.join(parts[6:])
                    })

        df = pd.DataFrame(data)
        return df

    for area in ['Mong','Chin']:
        end = datetime.datetime.now().strftime('%Y%m%d%H0000')
        start = (datetime.datetime.now()-datetime.timedelta(days=backwardsteps)).strftime('%Y%m%d%H0000')

        if area == 'Mong':
            df = quick_synop_test(start, end, area)
        else:
            df_n = quick_synop_test(start, end, area)
            df = pd.concat([df,df_n])

    # convert to numeric
    df['station'] = pd.to_numeric(df['station'], errors='coerce')
    df['station_clean'] = df['station'].astype(str).str.strip()
    df_stations['station_clean'] = df_stations['station'].astype(str).str.strip()
    df_selected = df[df['station_clean'].isin(df_stations['station_clean'])]
    def decode_synop_pressure(synop_code):
        try:
            result = s.SYNOP().decode(synop_code)
            pressure = result.get('sea_level_pressure', {}).get('value')
            return pressure
        except:
            return None

    df_selected['mlsp'] = df_selected['synop'].apply(decode_synop_pressure)
    df_selected = df_selected.drop(columns=['station_clean'])
    df_siberian = (
        df_selected
        .groupby(['year', 'month', 'day', 'hour'], as_index=False)
        .agg(avg_mlsp=('mlsp', lambda x: x.mean(skipna=True)))
    )

    df_siberian['time'] = df_siberian['year'] + '-' + df_siberian['month'] + '-' + df_siberian['day'] + ' ' + df_siberian['hour'] + ':00:00'

    def quick_synop_test(begin, end, wmoid):
        # URL dengan parameter spesifik
        url = "https://www.ogimet.com/cgi-bin/getsynop"
        params = {
            'begin': begin,  # 1 Des 2025 00:00
            'end': end,    # 1 Des 2025 23:00
            'block': wmoid
        }
        response = requests.get(url, params=params, timeout=10)

        # Parse ke DataFrame
        lines = response.text.strip().split('\n')
        data = []

        for line in lines:
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 7:
                    data.append({
                        'station': parts[0],
                        'year': parts[1],
                        'month': parts[2],
                        'day': parts[3],
                        'hour': parts[4],
                        'minute': parts[5],
                        'synop': ','.join(parts[6:])
                    })

        df = pd.DataFrame(data)
        return df


    for wmoid in ['58208','45007']:
        end = datetime.datetime.now().strftime('%Y%m%d%H0000')
        start = (datetime.datetime.now()-datetime.timedelta(days=backwardsteps)).strftime('%Y%m%d%H0000')

        if wmoid == '58208':
            df = quick_synop_test(start, end, wmoid)
            df['time'] = df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour'] + ':00:00'
        else:
            df_n = quick_synop_test(start, end, wmoid)
            df_n['time'] = df_n['year'] + '-' + df_n['month'] + '-' + df_n['day'] + ' ' + df_n['hour'] + ':00:00'
            df = df.merge(df_n, on=['time'])

    def decode_synop_pressure(synop_code):
        result = s.SYNOP().decode(synop_code)
        pressure = result.get('sea_level_pressure', {}).get('value')
        temperature = result.get('air_temperature', {}).get('value')
        return pressure,temperature

    hk_data = df['synop_y'].apply(decode_synop_pressure).apply(pd.Series)
    df[['mlsp_hk', 't2m_hk']] = hk_data.rename(columns={0: 'mlsp_hk', 1: 't2m_hk'})

    gs_data = df['synop_x'].apply(decode_synop_pressure).apply(pd.Series)
    df[['mlsp_gs', 't2m_gs']] = gs_data.rename(columns={0: 'mlsp_gs', 1: 't2m_gs'})

    df['mlsp_diff'] = df['mlsp_gs'] - df['mlsp_hk']

    # Pastikan time sebagai datetime dan set sebagai index
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(['station_x', 'time']).reset_index(drop=True)
    df = df.drop_duplicates(subset='time')

    def calculate_24h_diff(group):
        group = group.set_index('time')

        # Hapus duplikat index sebelum resample
        group = group[~group.index.duplicated(keep='first')]

        group_resampled = group[['t2m_hk']].resample('1H').interpolate(method='linear')
        group_resampled['t2m_hk_24h'] = group_resampled['t2m_hk'].diff(24)

        # Gabungkan kembali dengan data asli
        result = group_resampled.reindex(group.index)
        return result[['t2m_hk_24h']]

    df['t2m_hk_24h'] = df.groupby('station_x', group_keys=False).apply(lambda x: calculate_24h_diff(x)).values

    df_gshk = df[['time','mlsp_diff','t2m_hk_24h']]

    def scswind(time):
      timestr = time.strftime('%Y-%m-%d')
      hour = time.strftime('%H')
      if hour == '00':
        url = (f"https://weather.uwyo.edu/wsgi/sounding?datetime={timestr}%2{hour}:00:00&id=59981&src=UNKNOWN&type=TEXT:CSV")
      else:
        url = (f"https://weather.uwyo.edu/wsgi/sounding?datetime={timestr}%20{hour}:00:00&id=59981&src=UNKNOWN&type=TEXT:CSV")


      # --- Step 1: Ambil teks mentah ---
      response = requests.get(url)
      response.raise_for_status()
      text = response.text

      # --- Step 2: Cari baris header tabel ---
      lines = text.splitlines()

      header_line_idx = None
      for i, line in enumerate(lines):
          if line.startswith("PRES"):
              header_line_idx = i
              break

      # gabungkan list of strings menjadi satu teks CSV
      csv_text = "\n".join(lines)

      # baca sebagai DataFrame
      df = pd.read_csv(
          StringIO(csv_text),
          parse_dates=["time"]
      )

      # pastikan kolom numerik benar-benar numerik
      for col in df.columns:
          if col != "time":
              df[col] = pd.to_numeric(df[col], errors="coerce")

      # --- Step 5: Tangani missing value khas UWYO ---
      df.replace(
          to_replace=[9999, 99999, 999999],
          value=np.nan,
          inplace=True
      )

      theta = np.deg2rad(df["wind direction_degree"])
      V = df["wind speed_m/s"]
      df["u"] = -V * np.sin(theta)   # zonal (m/s)
      df["v"] = -V * np.cos(theta)   # meridional (m/s)
      df = df[df['pressure_hPa'] == 925][['time','u','v']]
      return df

    n = 0
    for dd in range(0,backwardsteps):
      for hh in [0,12]:
        try:
          time = (datetime.datetime.now()-datetime.timedelta(dd)).replace(hour=hh)
          if n==0:
            df = scswind(time)
            n+=1
          else:
            df_n = scswind(time)
            df = pd.concat([df,df_n])
            n+=1
        except:
          pass


    def replace_to_nearest_12_hour(dt_series):
        result = []
        for dt in dt_series:
            hour = dt.hour
            if hour < 6 or hour >= 18:
                new_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                if hour >= 18:
                    new_dt = new_dt + pd.Timedelta(days=1)
            else:
                new_dt = dt.replace(hour=12, minute=0, second=0, microsecond=0)
            result.append(new_dt)
        return pd.Series(result, index=dt_series.index)

    df['time'] = pd.to_datetime(df['time'])
    df['time_adjusted'] = replace_to_nearest_12_hour(df['time'])
    df_15N = df.sort_values(by='time')

    df_siberian['time'] = pd.to_datetime(df_siberian['time'])
    df_gshk['time'] = pd.to_datetime(df_gshk['time'])
    df_15N['time'] = pd.to_datetime(df_15N['time'])

    df = pd.merge(
        df_siberian[['time', 'avg_mlsp']],
        df_gshk[['time', 'mlsp_diff', 't2m_hk_24h']],
        on='time', how='inner'
    )
    df = pd.merge(
        df,
        df_15N[['time', 'v']],
        on='time', how='inner'
    )

    df_siberian['time'] = pd.to_datetime(df_siberian['time'])
    df_gshk['time'] = pd.to_datetime(df_gshk['time'])
    df_15N['time'] = pd.to_datetime(df_15N['time_adjusted'])

    do = pd.merge(
        df_siberian[['time', 'avg_mlsp']],
        df_gshk[['time', 'mlsp_diff', 't2m_hk_24h']],
        on='time', how= 'outer'
    )

    do = pd.merge(
        do,
        df_15N[['time', 'v']],
        on='time', how='outer'
    )

    cat = TDSCatalog(
        'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.xml'
    )
    cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                          'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')

    ncss = NCSS(cat.datasets[0].access_urls['NetcdfSubset'])
    # ncss.variables

    query = ncss.query()
    start = datetime.datetime.utcnow() - datetime.timedelta(days=backwardsteps)
    end   = datetime.datetime.utcnow()
    query.time_range(start, end)
    query.variables('v-component_of_wind_isobaric')
    query.vertical_level(92500)
    query.lonlat_box(north=17, south=-5, east=115, west=105)
    query.accept('netcdf')

    data = ncss.get_data(query)
    ds_wind = xr.open_dataset(
        xr.backends.NetCDF4DataStore(data)
    )

    # data = ncss.get_data(query)
    # nc = Dataset('inmemory.nc', memory=data)
    # ds_wind = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))

    query = ncss.query()
    start = datetime.datetime.utcnow() - datetime.timedelta(days=backwardsteps)
    end   = datetime.datetime.utcnow()
    query.time_range(start, end)
    query.variables('Temperature_height_above_ground', 'MSLP_Eta_model_reduction_msl')
    query.lonlat_box(north=60, south=22, east=116, west=80)
    query.accept('netcdf')
    # data = ncss.get_data(query)
    # nc = Dataset('inmemory.nc', memory=data)
    # ds_sfc = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))

    data = ncss.get_data(query)
    ds_sfc = xr.open_dataset(
        xr.backends.NetCDF4DataStore(data)
    )

    lat_hk = 22.30916667
    lon_hk = 113.9216667
    lat_gs = 32.25
    lon_gs = 115.6297222
    lat_15N = 16.830
    lon_15N = 112.330

    if 'time1' in ds_sfc.variables:
        time = ds_sfc['time1'].values
    else:
        time = ds_sfc['time'].values
    pred_siberian = (
        (ds_sfc['MSLP_Eta_model_reduction_msl'] / 100)
        .sel(
            longitude=xr.DataArray(df_stations['lon'].values, dims='station'),
            latitude=xr.DataArray(df_stations['lat'].values, dims='station'),
            method='nearest'
        )
        .mean(dim='station', skipna=True)
        .values
    )
    pred_phk = ds_sfc['MSLP_Eta_model_reduction_msl'].sel(longitude=lon_hk,latitude=lat_hk, method='nearest').values
    pred_pgs = ds_sfc['MSLP_Eta_model_reduction_msl'].sel(longitude=lon_gs,latitude=lat_gs, method='nearest').values
    pred_thk = ds_sfc['Temperature_height_above_ground'].sel(longitude=lon_hk,latitude=lat_hk, height_above_ground3=2, method='nearest').values
    pred_15N = ds_wind['v-component_of_wind_isobaric'].sel(longitude=lon_15N,latitude=lat_15N, method='nearest').isel(isobaric=0).values

    df_gfs = pd.DataFrame({'time':time, 'avg_mlsp':pred_siberian, 'pred_phk':pred_phk, 'pred_pgs':pred_pgs, 'pred_thk':pred_thk, 'v':pred_15N})
    df_gfs['t2m_hk_24h'] = (df_gfs['pred_thk']-273.15).diff(8)
    df_gfs['mlsp_diff'] = (df_gfs['pred_pgs'] - df_gfs['pred_phk'])/100
    df_gfs = df_gfs[['time','avg_mlsp','mlsp_diff','t2m_hk_24h','v']]

    cat = TDSCatalog(
        'https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.xml'
    )
    cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                          'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')

    ncss = NCSS(cat.datasets[0].access_urls['NetcdfSubset'])
    query = ncss.query()
    end = datetime.datetime.now() + datetime.timedelta(days=10)
    start   = datetime.datetime.now()
    query.time_range(start, end)
    query.variables('v-component_of_wind_isobaric')
    query.vertical_level(92500)
    query.lonlat_box(north=17, south=-5, east=115, west=105)
    query.accept('netcdf')
    # data = ncss.get_data(query)
    # nc = Dataset('inmemory.nc', memory=data)
    # ds_wind = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))

    data = ncss.get_data(query)
    ds_wind = xr.open_dataset(
        xr.backends.NetCDF4DataStore(data)
    )

    query = ncss.query()
    query.time_range(start, end)
    query.variables('Temperature_height_above_ground', 'MSLP_Eta_model_reduction_msl')
    query.lonlat_box(north=60, south=22, east=116, west=80)
    query.accept('netcdf')
    # data = ncss.get_data(query)
    # nc = Dataset('inmemory.nc', memory=data)
    # ds_sfc = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))

    data = ncss.get_data(query)
    ds_sfc = xr.open_dataset(
        xr.backends.NetCDF4DataStore(data)
    )

    if 'time1' in ds_sfc.variables:
        time = ds_sfc['time1'].values
    else:
        time = ds_sfc['time'].values
    pred_siberian = (
        (ds_sfc['MSLP_Eta_model_reduction_msl'] / 100)
        .sel(
            longitude=xr.DataArray(df_stations['lon'].values, dims='station'),
            latitude=xr.DataArray(df_stations['lat'].values, dims='station'),
            method='nearest'
        )
        .mean(dim='station', skipna=True)
        .values
    )
    pred_phk = ds_sfc['MSLP_Eta_model_reduction_msl'].sel(longitude=lon_hk,latitude=lat_hk, method='nearest').values
    pred_pgs = ds_sfc['MSLP_Eta_model_reduction_msl'].sel(longitude=lon_gs,latitude=lat_gs, method='nearest').values
    pred_thk = ds_sfc['Temperature_height_above_ground'].sel(longitude=lon_hk,latitude=lat_hk, height_above_ground3=2, method='nearest').values
    pred_15N = ds_wind['v-component_of_wind_isobaric'].sel(longitude=lon_15N,latitude=lat_15N, method='nearest').isel(isobaric=0).values

    df_pred = pd.DataFrame({'time':time, 'avg_mlsp':pred_siberian, 'pred_phk':pred_phk, 'pred_pgs':pred_pgs, 'pred_thk':pred_thk, 'v':pred_15N})
    df_pred['t2m_hk_24h'] = (df_pred['pred_thk']-273.15).diff(8)
    df_pred['mlsp_diff'] = (df_pred['pred_pgs'] - df_pred['pred_phk'])/100
    df_pred = df_pred[['time','avg_mlsp','mlsp_diff','t2m_hk_24h','v']]

    dobs = do.rename(columns={'avg_mlsp': 'obs_siberian',
                            'mlsp_diff': 'obs_gshk',
                            't2m_hk_24h': 'obs_thk_diff',
                            'v': 'obs_v15N'})

    df_gfs = df_gfs.rename(columns={'avg_mlsp': 'pred_siberian',
                            'mlsp_diff': 'pred_gshk',
                            't2m_hk_24h': 'pred_thk_diff',
                            'v': 'pred_v15N'})
    dc = dobs.merge(df_gfs, on='time', how='outer')
    # dc = dc.dropna()

    from scipy import interpolate
    import pandas as pd
    import numpy as np

    def quantile_mapping_with_mapping(obs_data, pred_data, new_pred_data=None,
                                      new_pred_mapping=None):
        """
        Quantile mapping dengan mapping eksplisit antar variabel

        Parameters:
        -----------
        obs_data : DataFrame dengan kolom observasi
        pred_data : DataFrame dengan kolom prediksi training
        new_pred_data : DataFrame dengan kolom prediksi baru
        new_pred_mapping : dict mapping dari pred_col ke new_pred_col

        Returns:
        --------
        DataFrame terkoreksi
        """
        # Mapping variabel asli
        variable_pairs = [
            ('obs_siberian', 'pred_siberian'),
            ('obs_gshk', 'pred_gshk'),
            ('obs_thk_diff', 'pred_thk_diff'),
            ('obs_v15N', 'pred_v15N')
        ]

        if new_pred_data is None:
            new_pred_data = pred_data
            new_pred_mapping = {pred_col: pred_col for _, pred_col in variable_pairs}

        # Jika mapping tidak diberikan, buat default
        if new_pred_mapping is None:
            # Default mapping untuk kasus Anda
            new_pred_mapping = {
                'pred_siberian': 'avg_mlsp',
                'pred_gshk': 'mlsp_diff',
                'pred_thk_diff': 't2m_hk_24h',
                'pred_v15N': 'v'
            }

        corrected_data = {}

        for obs_col, pred_col in variable_pairs:
            if obs_col in obs_data.columns and pred_col in pred_data.columns:
                # Ambil data training
                obs = obs_data[obs_col].values
                pred = pred_data[pred_col].values

                # Hapus NaN
                mask = ~(np.isnan(obs) | np.isnan(pred))
                obs_clean = obs[mask]
                pred_clean = pred[mask]

                if len(obs_clean) > 10:
                    # Hitung quantiles
                    n_q = min(100, len(obs_clean))
                    obs_q = np.percentile(obs_clean, np.linspace(0, 100, n_q))
                    pred_q = np.percentile(pred_clean, np.linspace(0, 100, n_q))

                    # Buat fungsi mapping
                    f = interpolate.interp1d(
                        pred_q, obs_q,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )

                    # Cari kolom yang sesuai di data baru
                    if pred_col in new_pred_mapping:
                        new_col = new_pred_mapping[pred_col]
                        if new_col in new_pred_data.columns:
                            corrected_data[pred_col] = f(new_pred_data[new_col].values)
                            print(f"✓ {obs_col} -> {pred_col} -> {new_col}")
                        else:
                            print(f"✗ Kolom {new_col} tidak ditemukan di new_pred_data")
                            # Gunakan nilai default
                            corrected_data[pred_col] = np.full(len(new_pred_data), np.nanmean(obs_clean))
                    else:
                        print(f"✗ Tidak ada mapping untuk {pred_col}")
                else:
                    print(f"✗ Data tidak cukup untuk {obs_col}")

        # Buat DataFrame
        df_corrected = pd.DataFrame(
            corrected_data,
            index=new_pred_data.index if hasattr(new_pred_data, 'index') else None
        )

        # Rename kolom untuk konsistensi
        df_corrected.columns = [col.replace('pred_', 'corrected_') for col in df_corrected.columns]

        return df_corrected

    # Definisikan mapping antara kolom training dan kolom baru
    pred_mapping = {
        'pred_siberian': 'avg_mlsp',
        'pred_gshk': 'mlsp_diff',
        'pred_thk_diff': 't2m_hk_24h',
        'pred_v15N': 'v'
    }

    # Koreksi data baru
    print("Melakukan quantile mapping...")
    df_new_corrected = quantile_mapping_with_mapping(
        dc[['obs_siberian','obs_gshk','obs_thk_diff','obs_v15N']],
        dc[['pred_siberian','pred_gshk','pred_thk_diff','pred_v15N']],
        df_pred[['avg_mlsp','mlsp_diff','t2m_hk_24h','v']],
        new_pred_mapping=pred_mapping
    )

    # Gabungkan dengan data asli
    df_pred_new = pd.concat([df_pred, df_new_corrected], axis=1)

    # Pilih dan rename kolom
    df_pred_new = df_pred_new[['time', 'corrected_siberian','corrected_gshk','corrected_thk_diff','corrected_v15N']]
    df_pred_new = df_pred_new.rename(columns={
        'corrected_siberian':'avg_mlsp',
        'corrected_gshk': 'mlsp_diff',
        'corrected_thk_diff': 't2m_hk_24h',
        'corrected_v15N': 'v'
    })

    # ------------------------------
    # 1. Persiapan data
    # ------------------------------
    df_interp = do.set_index('time')
    if len(df) > 1:
        freq_original = pd.infer_freq(df['time'])
        print(f"Frekuensi asli: {freq_original}")
    df_resampled = df_interp.resample('10min').asfreq()  # Buat grid waktu baru
    df_interpolated = df_resampled.interpolate(method='linear')  # Interpolasi linear
    df_interpolated = df_interpolated.reset_index()
    do = df_interpolated

    dp = df_pred_new

    # Thresholds
    THR_SO = 1045          # Siberian Outbreak
    THR_MSLP_DIFF = 10     # NCS arrival
    THR_TDROP = -5         # 24h drop (negatif: penurunan)

    THR_WEAK = -8
    THR_MOD = -10
    THR_STRONG = -12

    # Event Flags
    do['flag_SO'] = do['avg_mlsp'] > THR_SO
    do['flag_NCS_P'] = do['mlsp_diff'] > THR_MSLP_DIFF
    do['flag_NCS_T'] = do['t2m_hk_24h'] < THR_TDROP
    do['flag_NCS_SCS'] = do['v'] <= THR_WEAK

    dp['flag_SO'] = dp['avg_mlsp'] > THR_SO
    dp['flag_NCS_P'] = dp['mlsp_diff'] > THR_MSLP_DIFF
    dp['flag_NCS_T'] = dp['t2m_hk_24h'] < THR_TDROP
    dp['flag_NCS_SCS'] = dp['v'] <= THR_WEAK

    # ------------------------------
    # 2. Plotting
    # ------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    plt.suptitle("Northerly Cold Surge Diagnostic Time Series", fontweight='bold', fontsize=18, y=1.02)
    timestamp = datetime.datetime.now().strftime('%H:%MUTC %d-%m-%Y')
    fig.text(0.06, 0.98, f"Update: {timestamp}",
             fontsize=10, color='k',
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.))

    # ------------------------------
    # Panel 1 — avg_mlsp
    # ------------------------------
    axes[0].plot(do['time'], do['avg_mlsp'], label='Obs_avg_mlsp', linewidth=2)
    axes[0].axhline(THR_SO, color='red', linestyle='--', label='Siberian Outbreak Thresh (1045 mb)')

    # shading SO
    axes[0].fill_between(do['time'], do['avg_mlsp'], THR_SO,
                         where=do['flag_SO'],
                         color='red', alpha=0.2)
    # prediction
    axes[0].plot(dp['time'], dp['avg_mlsp'], label='Pred_avg_mlsp', linewidth=2, color='r')
    axes[0].fill_between(dp['time'], dp['avg_mlsp'], THR_SO,
                         where=dp['flag_SO'],
                         color='yellow', alpha=0.2)

    axes[0].set_ylabel("Siberian mean MSLP (mb)")
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(loc='upper left')
    axes[0].text(datetime.datetime.now()-datetime.timedelta(hours=213), do['avg_mlsp'].min(), f"Latest: {round(do['avg_mlsp'].iloc[-1],1)}", color='r', )

    # pred_siberian	pred_gshk	pred_thk_diff	pred_v15N

    # ------------------------------
    # Panel 2 — mlsp_diff
    # ------------------------------
    axes[1].plot(do['time'], do['mlsp_diff'], label='Obs_mlsp_diff (Gushi - HK)', linewidth=2)
    axes[1].axhline(THR_MSLP_DIFF, color='red', linestyle='--',
                    label='NCS Pressure Threshold (10 mb)')

    # shading NCS
    axes[1].fill_between(do['time'], do['mlsp_diff'], THR_MSLP_DIFF,
                         where=do['flag_NCS_P'],
                         color='red', alpha=0.2)

    #prediction
    axes[1].plot(dp['time'], dp['mlsp_diff'], label='Pred_mlsp_diff (Gushi - HK)', linewidth=2, color='r')
    axes[1].fill_between(dp['time'], dp['mlsp_diff'], THR_MSLP_DIFF,
                         where=dp['flag_NCS_P'],
                         color='yellow', alpha=0.2)


    axes[1].set_ylabel("MSLP diff (mb)")
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(loc='upper left')
    axes[1].text(datetime.datetime.now()-datetime.timedelta(hours=213), do['mlsp_diff'].min(), f"Latest: {round(do['mlsp_diff'].iloc[-1],1)}", color='r', )

    # ------------------------------
    # Panel 3 — t2m_hk_24h
    # ------------------------------
    axes[2].plot(do['time'], do['t2m_hk_24h'], label='Obs_24h Temp Drop HK', linewidth=2)
    axes[2].axhline(THR_TDROP, color='red', linestyle='--',
                    label='HK 24h Temp Drop Thresh (-5°C)')

    # shading NCS: gunakan flag_NCS
    axes[2].fill_between(do['time'], do['t2m_hk_24h'], THR_TDROP,
                         where=do['flag_NCS_T'],
                         color='red', alpha=0.2)

    #prediction
    axes[2].plot(dp['time'], dp['t2m_hk_24h'], label='Pred_24h Temp Drop HK', linewidth=2, color='r')
    axes[2].fill_between(dp['time'], dp['t2m_hk_24h'], THR_TDROP,
                         where=dp['flag_NCS_T'],
                         color='yellow', alpha=0.2)

    axes[2].set_ylabel("ΔT2m HK (°C/24h)")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, linestyle='--', alpha=0.4)
    axes[2].legend(loc='upper left')
    axes[2].text(datetime.datetime.now()-datetime.timedelta(hours=213), do['t2m_hk_24h'].min(), f"Latest: {round(do['t2m_hk_24h'].iloc[-1],1)}", color='r', )

    # ===== Panel 4 — NCS arrival in SCS (15N, 850 mb) =====
    axes[3].plot(do['time'], do['v'], lw=2, color='black', label='Obs |v| 925 mb (15°N)')
    axes[3].axhline(THR_WEAK, color='gold', ls='--', label='Weak (-8 to –10 m/s)')
    axes[3].axhline(THR_MOD, color='orange', ls='--', label='Moderate (-10 to –12 m/s)')
    axes[3].axhline(THR_STRONG, color='red', ls='--', label='Strong (<-12 m/s)')

    axes[3].fill_between(do['time'], do['v'], THR_WEAK,
                         where=(do['v'] <= THR_WEAK) & (do['v'] > THR_MOD),
                         color='gold', alpha=0.3)

    axes[3].fill_between(do['time'], do['v'], THR_MOD,
                         where=(do['v'] <= THR_MOD) & (do['v'] > THR_STRONG),
                         color='orange', alpha=0.3)

    axes[3].fill_between(do['time'], do['v'], THR_STRONG,
                         where=do['v'] <= THR_STRONG,
                         color='red', alpha=0.3)

    #prediction
    axes[3].plot(dp['time'], dp['v'], lw=2, label='Pred |v| 925 mb (15°N)', color='r')
    axes[3].fill_between(dp['time'], dp['v'], THR_WEAK,
                         where=(dp['v'] <= THR_WEAK) & (dp['v'] > THR_MOD),
                         color='gold', alpha=0.2)

    axes[3].fill_between(dp['time'], dp['v'], THR_MOD,
                         where=(dp['v'] <= THR_MOD) & (dp['v'] > THR_STRONG),
                         color='orange', alpha=0.2)

    axes[3].fill_between(dp['time'], dp['v'], THR_STRONG,
                         where=dp['v'] <= THR_STRONG,
                         color='red', alpha=0.2)


    axes[3].set_ylabel('|v| 925 mb (m/s)')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, ls='--', alpha=0.4)
    axes[3].legend(loc='upper left')
    axes[3].text(datetime.datetime.now()-datetime.timedelta(hours=213), do['v'].min(), f"Latest: {round(do['v'].iloc[-1],1)}", color='r', )

    plt.xlim(datetime.datetime.now()-datetime.timedelta(days=9),datetime.datetime.now()+datetime.timedelta(days=9))

    # Formatting x-axis
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)


