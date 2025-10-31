import os
import xarray as xr
import numpy as np
from collections import Counter

YEAR = ["2020", "2021", "2022", "2023", "2024"]
DATASET_ROOT = "E:/Study/5th Sem Mini Project/Datasets"
QUARTERS = ["Jan2Mar", "Apr2Jun", "Jul2Sept", "Oct2Dec"]
NC_FILENAMES = [
    "data_stream-oper_stepType-accum.nc",
    "data_stream-oper_stepType-instant.nc",
]


def to_0360(lon):
    return lon if lon >= 0 else lon + 360


def load_region_data(lat_min, lat_max, lon_min, lon_max, out_nc_path=None):
    lon_max = to_0360(lon_max)
    lon_min = to_0360(lon_min)

    def slice_dim(ds, dim, start, stop):
        coords = ds[dim].values
        if coords[0] > coords[-1]:
            return ds.sel({dim: slice(stop, start)})
        else:
            return ds.sel({dim: slice(start, stop)})

    all_var_datasets = []
    for year in YEAR:
        for quarter in QUARTERS:
            file_datasets = []
            for fname in NC_FILENAMES:
                fpath = os.path.join(DATASET_ROOT, year, quarter, fname)
                print(
                    f"Loading {fpath} for region latitude:[{lat_min},{lat_max}] longitude:[{lon_min},{lon_max}]"
                )
                ds = xr.open_dataset(fpath)
                ds_sel = ds.pipe(slice_dim, "latitude", lat_min, lat_max)
                ds_sel = ds_sel.pipe(slice_dim, "longitude", lon_min, lon_max)
                ds_sel = ds_sel.drop_vars("expver", errors="ignore")
                file_datasets.append(ds_sel)
            quarter_combined = xr.merge(file_datasets, compat="override")
            all_var_datasets.append(quarter_combined)

    data_combined = xr.concat(all_var_datasets, dim="valid_time").sortby("valid_time")
    print("Data for region combined shape:", data_combined.sizes)
    if out_nc_path is not None:
        print(f"Saving combined dataset to: {out_nc_path}")
        data_combined.to_netcdf(out_nc_path)
    return data_combined


META_TRAIN_REGIONS = [
    (-9.5, -4.5, -67.5, -62.5),
    (12.5, 17.5, 102.5, 107.5),
    (22.5, 27.5, 19.5, 24.5),
    (-23.5, -18.5, 132.5, 137.5),
    (43.5, 48.5, 7.5, 12.5),
    (35.5, 40.5, -5.5, -0.5),
    (53.5, 58.5, 34.5, 39.5),
    (44.5, 49.5, 125.5, 130.5),
    (67.5, 72.5, -32.5, -27.5),
    (-20, -15, -70, -65),
    (32.5, 37.5, 137.5, 142.5),
    (-35.5, -30.5, 16.5, 21.5),
    (51.5, 56.5, -112.5, -107.5),
    (29.5, 34.5, -101.5, -96.5),
    (11.5, 16.5, 86.5, 91.5),
]

# Koppen codes and corresponding classes from your metadata
code_to_class = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BSh",
    5: "BSk",
    6: "BWh",
    7: "BWk",
    8: "Cfa",
    9: "Cfb",
    10: "Cfc",
    11: "Csa",
    12: "Csb",
    13: "Csc",
    14: "Cwa",
    15: "Cwb",
    16: "Cwc",
    17: "Dfa",
    18: "Dfb",
    19: "Dfc",
    20: "Dfd",
    21: "Dsa",
    22: "Dsb",
    23: "Dsc",
    24: "Dsd",
    25: "Dwa",
    26: "Dwb",
    27: "Dwc",
    28: "Dwd",
    29: "EF",
    30: "ET",
}


def get_koppen_class(lat_min, lat_max, lon_min, lon_max):
    ds = xr.open_dataset("E:/Study/5th Sem Mini Project/Datasets/RobustKGMaps.nc")

    def slice_dim(ds, dim, start, stop):
        coords = ds[dim].values
        if coords[0] > coords[-1]:
            return ds.sel({dim: slice(stop, start)})
        else:
            return ds.sel({dim: slice(start, stop)})

    ds_sel = ds.pipe(slice_dim, "lat", lat_min, lat_max)
    ds_sel = ds_sel.pipe(slice_dim, "lon", lon_min, lon_max)

    koppen_data = ds_sel["MasterMap1"].values.flatten()

    koppen_data = koppen_data[~np.isnan(koppen_data)].astype(int)

    if len(koppen_data) == 0:
        return -1  # No data in region

    counts = Counter(koppen_data)
    majority_code = counts.most_common(1)[0][0]

    return majority_code


def main_dataloader(lat_min, lat_max, lon_min, lon_max):
    out_nc_path = f"E:/Study/5th Sem Mini Project/Code/Out_Data/lat{lat_min}-{lat_max}_lon{lon_min}-{lon_max}.nc"
    return (
        load_region_data(lat_min, lat_max, lon_min, lon_max, out_nc_path=out_nc_path),
        get_koppen_class(lat_min, lat_max, lon_min, lon_max),
        out_nc_path,
    )
