# Base selectors
base_selectors = ["system:index", "lat", "lon", "id", "date"]

# Sentinel-1 selectors
s1_selectors = ["LIA", "VH", "VV", "VVVH_ratio", "angle"]

# GLDAS selectors
gldas_selectors = [
    "sm_1",
    "sm_1_100",
    "sm_1_40",
    "sm_3",
    "sm_3_100",
    "sm_3_40",
    "sm_7",
    "sm_7_100",
    "sm_7_40",
    "sm_30",
    "sm_30_100",
    "sm_30_40",
]

# GPM selectors
gpm_selectors = ["precipitation", "prec_3", "prec_7", "prec_30"]
gpm_selectors_sum = ["prec_3_sum", "prec_7_sum", "prec_30_sum"]

# Hansen selectors (excluding 'year' for explanatory variables)
hansen_selectors = ["year", "B3", "B4", "B5", "B7", "ndvi", "ndmi", "ndbri"]

# Temporal variables
temporal_vars = (
    ["doy"]  # Day of year
    + s1_selectors  # Sentinel-1 temporal variables
    + gldas_selectors  # GLDAS temporal variables
    + gpm_selectors  # GPM temporal variables
)

# Non-temporal variables
non_temporal_vars = (
    ["elevation", "aspect", "slope"]  # Topographic variables
    + ["land_cov", "canopy_height"]  # Land cover variables
    + ["gldas_mean", "gldas_stddev"]  # GLDAS statistical variables
    + ["distance", "dir", "acc", "land_forms"]  # Additional variables
)

# Explanatory variables
explain_vars = temporal_vars + hansen_selectors[1:] + non_temporal_vars

# Response variable
response_var = ["gwl_cm"]

# Temporal explanatory variables subset
temporal_expl = (
    ["VV", "VH", "VVVH_ratio"]  # Sentinel-1 variables
    + ["precipitation", "prec_3", "prec_7", "prec_30"]  # GPM variables
    + ["sm_1", "sm_3", "sm_7", "sm_30"]  # GLDAS variables
)

# Regions IDs
regions_ids = {
    "sumatra": [1, 2, 4, 5, 6],
    "kalimantan": [7, 8, 10],
    "eastern": [3, 9],
    "all": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

# Bad stations
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

# Best PHUs in Kalimantan
best_kalimantan_phus = [350, 351, 357, 379]
