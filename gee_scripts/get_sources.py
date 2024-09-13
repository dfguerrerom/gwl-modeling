from eeSAR.s1 import s1_collection
from datetime import datetime as dt, timedelta

import ee
import pandas as pd


def get_s1_dates(aoi) -> list:
    """Get the dates of the Sentinel-1 images for a given location."""

    orbits = ["ASCENDING", "DESCENDING"]

    # get the image
    return (
        (
            ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
            .filterBounds(aoi)
            .filterMetadata("resolution_meters", "equals", 10)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(
                ee.Filter.And(
                    ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
                    ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
                )
            )
            .filter(
                ee.Filter.Or(
                    [ee.Filter.eq("orbitProperties_pass", orbit) for orbit in orbits]
                )
            )
        ).aggregate_array("system:time_start")
        # .filter(ee.Filter.gt("item", 1546300800000))  # 2019-01-01
        .getInfo()
    )


def get_s1_str_dates(target_phu: ee.Geometry, start_date: str, end_date: str) -> list:

    # Get all the dates of the Sentinel 1 images for the target phu
    all_s1_dates = get_s1_dates(aoi=target_phu)

    # Convert the dates to a string format
    all_str_date = [
        dt.fromtimestamp(date / 1000).strftime("%Y-%m-%d") for date in all_s1_dates
    ]

    dates_df = pd.DataFrame(all_str_date, columns=["date"])
    # drop duplicates
    dates_df = dates_df.drop_duplicates()
    dates_df["date"] = pd.to_datetime(dates_df["date"])

    dates = dates_df[(dates_df["date"] >= start_date) & (dates_df["date"] <= end_date)]

    all_str_date = dates["date"].dt.strftime("%Y-%m-%d").tolist()

    return all_str_date


def add_date_difference(image, t):
    return image.set(
        "dateDist",
        ee.Number(image.get("system:time_start")).subtract(t.millis()).abs(),
    )


def set_resample(image):
    """Set resampling of the image to bilinear"""
    return image.resample()


def get_gldas(date: ee.DateRange, aoi: ee.Geometry) -> ee.Image:
    """Get the GLDAS image for a given date and location."""

    def set_resample(image):
        """Set resampling of the image to bilinear"""
        return image.resample()

    to = date.start()
    from_ = to.advance(ee.Number(-30), "days")

    gldas = (
        ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
        .select("SoilMoi0_10cm_inst")
        .filterBounds(aoi)
        .filterDate(from_, to)
        # .map(set_resample)
        .map(lambda img: add_date_difference(img, to))
    )

    sm_gldas = gldas.sort("dateDist").first().rename("sm_1")

    gldas_3day = gldas.filterDate(to.advance(ee.Number(-3), "days"), to)
    gldas_3day = gldas_3day.sum().divide(gldas_3day.count()).rename("sm_3")

    gldas_7day = gldas.filterDate(to.advance(ee.Number(-7), "days"), to)
    gldas_7day = gldas_7day.sum().divide(gldas_7day.count()).rename("sm_7")

    gldas_30day = gldas.sum().divide(gldas.count()).rename("sm_30")

    return sm_gldas.addBands(gldas_3day).addBands(gldas_7day).addBands(gldas_30day)


def get_gpm(date: ee.DateRange, aoi: ee.Geometry) -> ee.Image:
    """Get the GPM image for a given date and location."""
    to = date.start()
    from_ = to.advance(ee.Number(-30), "days")

    gpm = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterBounds(aoi)
        .filterDate(from_, to)
        .select("HQprecipitation")
        # .map(set_resample)
        .map(lambda img: add_date_difference(img, to))
    )

    gpm_closest = gpm.filterDate(to.advance(ee.Number(-1), "days"), to)
    gpm_closest = gpm_closest.sum().divide(gpm_closest.count()).rename("precipitation")

    gpm_3day = gpm.filterDate(to.advance(ee.Number(-3), "days"), to)
    gpm_3day = gpm_3day.sum().divide(gpm_3day.count()).rename("prec_3")

    gpm_7day = gpm.filterDate(to.advance(ee.Number(-7), "days"), to)
    gpm_7day = gpm_7day.sum().divide(gpm_7day.count()).rename("prec_7")

    gpm_30day = gpm.sum().divide(gpm.count()).rename("prec_30")

    return gpm_closest.addBands(gpm_3day).addBands(gpm_7day).addBands(gpm_30day)


def get_gpm_sum(date: ee.DateRange, aoi: ee.Geometry.Point) -> ee.Image:
    """Get the GPM image for a given date and location but return the sum instead of the mean."""
    to = date.start()
    from_ = to.advance(ee.Number(-30), "days")

    gpm = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterBounds(aoi)
        .filterDate(from_, to)
        .select("HQprecipitation")
        # .map(set_resample)
        .map(lambda img: add_date_difference(img, to))
    )

    gpm_3day = gpm.filterDate(to.advance(ee.Number(-3), "days"), to)
    gpm_3day = gpm_3day.sum().rename("prec_3_sum")

    gpm_7day = gpm.filterDate(to.advance(ee.Number(-7), "days"), to)
    gpm_7day = gpm_7day.sum().rename("prec_7_sum")

    gpm_30day = gpm.sum().rename("prec_30_sum")

    return gpm_3day.addBands(gpm_7day).addBands(gpm_30day)


def get_srtm() -> ee.Image:
    """Returns the SRTM image."""
    srtm = ee.Image("USGS/SRTMGL1_003").resample()
    aspect = ee.Terrain.aspect(srtm).rename("aspect")
    slope = ee.Terrain.slope(srtm).rename("slope")

    return srtm.select("elevation").addBands(aspect).addBands(slope)


def get_globcover() -> ee.Image:
    """Returns the GlobCover image for 2020."""

    return ee.Image("users/amitghosh/sdg_module/esa/cci_landcover/2020").select(
        [0], ["land_cov"]
    )


def get_s1_collection(date: ee.DateRange, aoi: ee.Geometry.Point) -> ee.Image:
    """Get the Sentinel-1 collection for a given date and location."""
    image = s1_collection.create(
        region=aoi,
        start_date=date.start(),
        end_date=date.end(),
        add_ND_ratio=False,
        speckle_filter="QUEGAN",
        radiometric_correction="TERRAIN",
        slope_correction_dict={
            "model": "surface",
            "dem": "USGS/SRTMGL1_003",
            "buffer": 50,
        },  #'CGIAR/SRTM90_V4'
        db=True,
    )

    return image


def get_tsscans(date: str, aoi: ee.Geometry) -> ee.Image:
    """Get the timescans for a given date and location."""

    def toLn(image):
        ln = image.select(["VV", "VH"]).log().rename(["VV", "VH"])
        return image.addBands(ln, None, True)

    def toLin(image):
        lin = (
            ee.Image(10).pow(image.select(["VV", "VH"]).divide(10)).rename(["VV", "VH"])
        )
        return image.addBands(lin, None, True)

    image = get_s1_collection(date, aoi).median()

    track_nr = image.get("relativeOrbitNumber_start")
    orbit = image.get("orbitProperties_pass").getInfo()

    tSeries = s1_collection.create(
        region=image.geometry(),
        orbits=[orbit],
        start_date="2018-01-01",
        end_date="2022-12-31",
        add_ratio=False,
        add_ND_ratio=False,
        speckle_filter="NONE",
        radiometric_correction="TERRAIN",
        slope_correction_dict={
            "model": "surface",
            "dem": "USGS/SRTMGL1_003",
            "buffer": 50,
        },
        db=False,
        outlier_removal="AGGRESSIVE",
    ).filterMetadata("relativeOrbitNumber_start", "equals", track_nr)

    # create combined reducer
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.percentile([5, 95]), "", True)
    )

    # create log timescan (k variables)
    tScanLn = (
        tSeries.map(toLn)
        .select(["VV", "VH"])
        .reduce(reducer)
        .select(
            [
                "VV_mean",
                "VV_stdDev",
                "VV_p5",
                "VV_p95",
                "VH_mean",
                "VH_stdDev",
                "VH_p5",
                "VH_p95",
            ],
            [
                "kVV_mean",
                "kVV_stdDev",
                "kVV_p5",
                "kVV_p95",
                "kVH_mean",
                "kVH_stdDev",
                "kVH_p5",
                "kVH_p95",
            ],
        )
    )

    # creeate linear timescan
    tScanLin = tSeries.map(toLin).select(["VV", "VH"]).reduce(reducer)

    return image.addBands(tScanLn).addBands(tScanLin)


def get_gldas_stats(aoi: ee.Geometry) -> ee.Image:
    """Generate the GLDAS mean and stddev image for a given location."""

    gldas = (
        ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
        .select("SoilMoi0_10cm_inst")
        .filterBounds(aoi)
        .filterDate("2018-01-01", "2022-12-31")
        .map(set_resample)
    )

    return gldas.reduce(
        ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)
    ).rename("gldas_mean", "gldas_stddev")


def get_gedi(aoi: ee.Geometry) -> ee.Image:
    """Get the GEDI image for a given date and location."""

    return (
        ee.ImageCollection("users/potapovpeter/GEDI_V27")
        .filterBounds(aoi)
        .first()
        .rename("canopy_height")
    )


def get_hansen(year):
    """returns the hansen asset id based on the year"""

    hansen_start_year = 2012
    last_hansen_year = 2023

    # use the latest hansen data if the year is not available
    year = 2022 if year >= last_hansen_year else year

    # we start from version 1.0 for 2012
    version = year - hansen_start_year
    hansen = ee.Image(f"UMD/hansen/global_forest_change_{year}_v1_{version}")

    b3 = hansen.select(["last_b30"], ["B3"])
    b4 = hansen.select(["last_b40"], ["B4"])
    b5 = hansen.select(["last_b50"], ["B5"])
    b7 = hansen.select(["last_b70"], ["B7"])

    ndvi = (b4.subtract(b3)).divide(b4.add(b3)).rename("ndvi")
    ndmi = (b4.subtract(b5)).divide(b4.add(b5)).rename("ndmi")
    ndbri = (b4.subtract(b7)).divide(b4.add(b7)).rename("ndbri")

    return (
        ee.Image()
        .addBands(b3)
        .addBands(b4)
        .addBands(b5)
        .addBands(b7)
        .addBands(ndvi)
        .addBands(ndmi)
        .addBands(ndbri)
    )


def add_diff(image, target_date: str):
    diff = (
        ee.Number(image.date().millis())
        .subtract(ee.Number(ee.Date(target_date).millis()))
        .abs()
    )
    return image.set("diff", diff)


def get_extra_non_temporal():
    """Returns the distance, dir and acc bands."""

    water = ee.FeatureCollection(
        "users/marortpab/FAO/SEPAL/2023_trainings/smm/water_bodies_phu_buff_1_km_def"
    )
    canals = ee.FeatureCollection(
        "users/marortpab/FAO/SEPAL/2023_trainings/smm/prims_canal_data"
    )

    hydro_dir = ee.Image("WWF/HydroSHEDS/03DIR")
    drain_direction = hydro_dir.select("b1").rename("dir")
    hydro_acc = ee.Image("WWF/HydroSHEDS/15ACC")
    flow_acc = hydro_acc.select("b1").rename("acc")
    land_forms = ee.Image("CSP/ERGo/1_0/Global/SRTM_landforms").rename("land_forms")

    # //Merge water bodies and canals

    all_water = water.merge(canals)
    distance = all_water.distance(25000)

    return distance.addBands(drain_direction).addBands(flow_acc).addBands(land_forms)


def get_explanatory_composite(target_date: str, ee_region: ee.Geometry):
    """Get the closest explanatory image to the target date

    Args:
        target_date: date to get the image for
        ee_region: region to get the image for
    """

    # search range
    year = dt.strptime(target_date, "%Y-%m-%d").year

    # get start and end date using dt
    start_date = dt.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)
    end_date = dt.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)

    # if end_date is in the future, set it to today
    if end_date > dt.now():
        end_date = dt.now()

    # Create a gee daterange object
    date_range = ee.DateRange(start_date, end_date)

    # For each image in the image collection, add a new property called 'diff'
    # which is the difference between the image timestamp and the target date

    # Map the function over the image collection
    s1_composite = s1_collection.create(
        region=ee_region,
        start_date=date_range.start(),
        end_date=date_range.end(),
        add_ND_ratio=False,
        speckle_filter="QUEGAN",
        radiometric_correction="TERRAIN",
        slope_correction_dict={
            "model": "surface",
            "dem": "USGS/SRTMGL1_003",
            "buffer": 50,
        },  #'CGIAR/SRTM90_V4'
        db=True,
    ).select(["LIA", "VH", "VV", "VVVH_ratio", "angle"])

    # Get the image timestamp
    s1_date = (
        ee.Date(s1_composite.first().get("system:time_start"))
        .format("YYYY-MM-dd")
        .getInfo()
    )

    # Composite all the images in the search range
    s1_composite = s1_composite.median()

    date_range = ee.Date(target_date).getRange("day")
    gldas_image = get_gldas(date_range, ee_region)
    gpm_iamge = get_gpm(date_range, ee_region)
    gpm_sum = get_gpm_sum(date_range, ee_region)

    # create date of the year from string datetime
    date = dt.strptime(s1_date, "%Y-%m-%d")
    doy = date.timetuple().tm_yday

    # Extract the year of the target date
    composite = (
        s1_composite.addBands(gldas_image)
        .addBands(gpm_iamge)
        .addBands(gpm_sum)
        .addBands(get_hansen(year))
        .addBands(get_srtm())
        .addBands(get_globcover())
        .addBands(get_gedi(ee_region))
        .addBands(get_gldas_stats(ee_region))
        .addBands(ee.Image.constant(doy).rename(["doy"]))
        .addBands(get_extra_non_temporal())
        .toFloat()
    ).set({"system:time_start": ee.Date(s1_date)})

    print("Composite for date: ", s1_date, " created.")

    return composite
