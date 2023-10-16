from eeSAR.s1 import s1_collection
from datetime import datetime as dt, timedelta

import ee

ee.Initialize()

from typing import Union


def add_date_difference(image, t):
    return image.set(
        "dateDist",
        ee.Number(image.get("system:time_start")).subtract(t.millis()).abs(),
    )


def set_resample(image):
    """Set resampling of the image to bilinear"""
    return image.resample()


def is_s1_image(date: str, aoi: ee.Geometry) -> Union[None, ee.Image]:
    """Check if there is a Sentinel-1 image for a given date and location.

    The different filters are based on the eeSar.s1.s1_collection.create() function.
    """

    orbits = ["ASCENDING", "DESCENDING"]
    start_date = dt.strptime(date, "%Y-%m-%d")
    end_date = start_date + timedelta(days=1)

    # get the image
    image = (
        ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
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
    ).getInfo()

    return True if image["features"] else False


def get_gldas(date: str, aoi: ee.Geometry) -> ee.Image:
    """Get the GLDAS image for a given date and location."""

    def set_resample(image):
        """Set resampling of the image to bilinear"""
        return image.resample()

    t = ee.Date(date)
    fro = t.advance(ee.Number(-30), "days")
    # to = t.advance(ee.Number(10), 'days')

    gldas = (
        ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
        .select("SoilMoi0_10cm_inst")
        .filterBounds(aoi)
        .map(set_resample)
    )
    gldas_stat = gldas.reduce(
        ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)
    ).rename("gldas_mean", "gldas_stddev")

    gldas = gldas.filterDate(fro, t).map(lambda img: add_date_difference(img, t))

    sm_gldas = gldas.sort("dateDist").first().rename("sm_1")

    gldas_3day = gldas.filterDate(t.advance(ee.Number(-3), "days"), t)
    gldas_3day = gldas_3day.sum().divide(gldas_3day.count()).rename("sm_3")

    gldas_7day = gldas.filterDate(t.advance(ee.Number(-7), "days"), t)
    gldas_7day = gldas_7day.sum().divide(gldas_7day.count()).rename("sm_7")

    gldas_30day = gldas.sum().divide(gldas.count()).rename("sm_30")

    return (
        gldas_stat.addBands(sm_gldas)
        .addBands(gldas_3day)
        .addBands(gldas_7day)
        .addBands(gldas_30day)
    )


def get_gpm(date: str, aoi: ee.Geometry) -> ee.Image:
    t = ee.Date(date)
    # t = ee.Date(feature.get('date').getInfo()['value'])
    fro = t.advance(ee.Number(-30), "days")

    gpm = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
        .filterBounds(aoi)
        .filterDate(fro, t)
        .select("HQprecipitation")
        .map(lambda img: add_date_difference(img, t))
        .map(set_resample)
    )

    gpm_closest = gpm.filterDate(t.advance(ee.Number(-1), "days"), t)
    gpm_closest = gpm_closest.sum().divide(gpm_closest.count()).rename("precipitation")

    gpm_3day = gpm.filterDate(t.advance(ee.Number(-3), "days"), t)
    gpm_3day = gpm_3day.sum().divide(gpm_3day.count()).rename("prec_3")

    gpm_7day = gpm.filterDate(t.advance(ee.Number(-7), "days"), t)
    gpm_7day = gpm_7day.sum().divide(gpm_7day.count()).rename("prec_7")

    gpm_30day = gpm.sum().divide(gpm.count()).rename("prec_30")

    return gpm_closest.addBands(gpm_3day).addBands(gpm_7day).addBands(gpm_30day)


def get_srtm(date: str, aoi: ee.Geometry) -> ee.Image:
    srtm = ee.Image("USGS/SRTMGL1_003").resample()
    aspect = ee.Terrain.aspect(srtm).rename("aspect")
    slope = ee.Terrain.slope(srtm).rename("slope")

    return srtm.select("elevation").addBands(aspect).addBands(slope)


def get_globcover(date: str, aoi: ee.Geometry) -> ee.Image:
    return ee.Image("ESA/GLOBCOVER_L4_200901_200912_V2_3")


def get_s1_image(date: str, aoi: ee.Geometry.Point) -> ee.Image:
    """Get the Sentinel-1 image for a given date and location."""

    start = dt.strptime(date, "%Y-%m-%d")
    end = start + timedelta(days=1)

    image = s1_collection.create(
        region=aoi.buffer(2000),
        start_date=start,
        end_date=end,
        add_ND_ratio=False,
        speckle_filter="QUEGAN",
        radiometric_correction="TERRAIN",
        slope_correction_dict={
            "model": "surface",
            "dem": "USGS/SRTMGL1_003",
            "buffer": 50,
        },  #'CGIAR/SRTM90_V4'
        db=True,
    ).first()

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

    image = get_s1_image(date, aoi)

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


def get_gedi(date: str, aoi: ee.Geometry) -> ee.Image:
    """Get the GEDI image for a given date and location."""

    return (
        ee.ImageCollection("users/potapovpeter/GEDI_V27")
        .mosaic()
        .clip(aoi)
        .rename("canopy_height")
    )


def get_hansen(date: str, *_) -> ee.Image:
    """Get the Hansen image for a given date and location."""

    start_year = 2012

    year = dt.strptime(date, "%Y-%m-%d").year
    year = 2022 if year >= 2023 else year

    # we start from version 1.0 for 2012
    version = year - start_year

    hansen = ee.Image(f"UMD/hansen/global_forest_change_{year}_v1_{version}")

    b3 = hansen.select(["last_b30"], ["B3"])
    b4 = hansen.select(["last_b40"], ["B4"])
    b5 = hansen.select(["last_b50"], ["B5"])
    b7 = hansen.select(["last_b70"], ["B7"])

    ndvi = (b4.subtract(b3)).divide(b4.add(b3)).rename("ndvi")
    ndmi = (b4.subtract(b5)).divide(b4.add(b5)).rename("ndmi")
    ndbri = (b4.subtract(b7)).divide(b4.add(b7)).rename("ndbri")

    return (
        b3.addBands(b4)
        .addBands(b5)
        .addBands(b7)
        .addBands(ndvi)
        .addBands(ndmi)
        .addBands(ndbri)
    )
