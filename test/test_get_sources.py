"""Test scripts in gee_scripts/get_sources.py"""

import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

import ee
import pytest
from gee_scripts.get_sources import get_gldas


def test_get_gldas():

    test_point = {
        "id": "NASAKF-01D10KF-01-D-10",
        "date": "2016-02-12",
        "latitude": -2.29661,
        "longitude": 114.52164,
        "gwl_cm": -6.2,
        "expected_results": {
            "sm_1": 37.064998626708984,
            "sm_1_100": 223.05799865722656,
            "sm_1_40": 111.63837432861328,
            "sm_3": 37.8307081858317,
            "sm_30": 35.46240828831991,
            "sm_30_100": 206.0546452840169,
            "sm_30_40": 105.91919797261556,
            "sm_3_100": 224.76087379455566,
            "sm_3_40": 113.7289244333903,
            "sm_7": 37.82410710198538,
            "sm_7_100": 220.83594621930803,
            "sm_7_40": 113.08218056815011,
        },
    }

    start_date = ee.Date(test_point["date"])
    end_date = ee.Date(test_point["date"]).advance(1, "day")
    date_range = ee.DateRange(start_date, end_date)

    aoi = ee.Geometry.Point(test_point["longitude"], test_point["latitude"])

    gldas = get_gldas(date_range, aoi)

    assert sorted(gldas.bandNames().getInfo()) == sorted(
        list(test_point["expected_results"].keys())
    )

    # Reduce by point and get the first feature

    result = gldas.reduceRegions(
        **{
            "collection": ee.FeatureCollection([ee.Feature(aoi)]),
            "reducer": ee.Reducer.first(),
            "scale": 10,
            "tileScale": 16,
        }
    )

    assert (
        result.getInfo()["features"][0]["properties"] == test_point["expected_results"]
    )


if __name__ == "__main__":
    # Run pytest with the current file
    pytest.main([__file__, "-s", "-vv"])

[
    "sm_1",
    "sm_1_100",
    "sm_1_40",
    "sm_3",
    "sm_30",
    "sm_30_100",
    "sm_30_40",
    "sm_3_100",
    "sm_3_40",
    "sm_7",
    "sm_7_100",
    "sm_7_40",
]
