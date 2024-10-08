explain_vars = [
    "doy",
    "LIA",
    "VH",
    "VV",
    "VVVH_ratio",
    "angle",
    "sm_1",
    "sm_3",
    "sm_7",
    "sm_30",
    "precipitation",
    "prec_3",
    "prec_7",
    "prec_30",
    "elevation",
    "aspect",
    "slope",
    "land_cov",
    "canopy_height",
    "gldas_mean",
    "gldas_stddev",
    "B3",
    "B4",
    "B5",
    "B7",
    "ndvi",
    "ndmi",
    "ndbri",
    "distance",
    "dir",
    "acc",
    "prec_3_sum",
    "prec_7_sum",
    "prec_30_sum",
    # "land_forms",
]

biophysical_vars = [
    "elevation",
    "aspect",
    "slope",
    "land_cov",
    "canopy_height",
    "distance",
    "dir",
    "acc",
    "land_forms",
]

response_var = ["gwl_cm"]

temporal_expl = [
    "VV",
    "VH",
    "VVVH_ratio",
    "precipitation",
    "prec_3",
    "prec_7",
    "prec_30",
    "sm_1",
    "sm_3",
    "sm_7",
    "sm_30",
]

regions_ids = {
    "sumatra": [1, 6, 5, 2, 4],
    "kalimantan": [7, 8, 10],
    "estern": [3, 9],
    "all": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


bad_stations = [
    "batok1",
    "batok2",
    "brg11",
    "brg13",
    "brg16",
    "BRG_620309_01",
    "BRG_620309_02",
    "BRG_630805_01",
    "BRG_630708_01",
]
