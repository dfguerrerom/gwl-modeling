import re
import os
import rasterio
from pyproj import Proj, transform
from pyproj.transformer import Transformer
from pathlib import Path
import pickle
from shapely.geometry import Point
from datetime import datetime


def find_pixel_values_for_coordinates(index, coords):
    results = []
    i = 0
    for filepath, meta in index.items():
        bounds, _ = meta["bounds"], meta["crs"]

        relevant_coords = [
            ((x, y), point_id)
            for (x, y), point_id in coords
            if Point(x, y).within(bounds)
        ]

        if not relevant_coords:
            continue

        with rasterio.open(filepath) as src:
            for (x, y), point_id in relevant_coords:
                row, col = src.index(x, y)
                value = src.read(1)[row, col]
                results.append(
                    {
                        "image": filepath,
                        "smm_value": value,
                        "coordinate": (x, y),
                        "date": get_date(filepath),
                        "point_id": point_id,
                    }
                ),
            i += 1

        if i % 2000 == 0:
            print(f"Processed {i} images")

    return results


def get_image_index(all_images: list):
    """Create an index of all images so we don't have to generate this process again"""

    index = {}
    i = 0
    for image in all_images:
        with rasterio.open(image) as src:
            bounds = src.bounds
            crs = src.crs
            index[image] = {"bounds": Polygon(box(*bounds)), "crs": crs}
            i += 1
        if i % 1000 == 0:
            print(f"Processed {i}")

    return index


def get_date(image_name):
    """Extract the date of a given image

    Expected patter: nclose_SMCmap_2019_11_09_dguerrero_1111.tif

    """
    filename = Path(image_name).stem
    match = re.search(r"\d{4}_\d{2}_\d{2}", filename)
    date = datetime.strptime(match.group(), "%Y_%m_%d").date()

    return date


def test_find_pixel_values_for_coordinates(index):
    test_coords = [
        ((109.896029, -0.443315), "10_MTI_MTI_J127_P4"),
        ((109.934069, -0.482402), "10_MTI_MTI_K085_P4"),
        ((133.054459, -2.909185), "271_RSP_H42"),
    ]

    expected_results = [
        {
            "image": "/home/sepal-user/soil_moisture/papua_dan/close_SMCmap_2020_06_21_sepal-user_sample_3_ALL_chip_0.tif",
            "value": 26,
            "id": "271_RSP_H42",
            "date": datetime(2020, 6, 21).date(),
            "coordinate": (133.054459, -2.909185),
        },
        {
            "image": "/home/sepal-user/soil_moisture/kalimantan_island/close_SMCmap_2022_01_30_sepal-user_PHU_883_group_08_phu_id_85chip_0.tif",
            "value": 20,
            "id": "10_MTI_MTI_K085_P4",
            "date": datetime(2022, 1, 30).date(),
            "coordinate": (109.934069, -0.482402),
        },
        {
            "image": "/home/sepal-user/soil_moisture/kalimantan_island/close_SMCmap_2020_10_07_sepal-user_PHU_883_group_08_phu_id_85chip_0.tif",
            "value": 17,
            "id": "10_MTI_MTI_K085_P4",
            "date": datetime(2020, 10, 7).date(),
            "coordinate": (109.934069, -0.482402),
        },
        {
            "image": "/home/sepal-user/soil_moisture/papua_dan/close_SMCmap_2021_07_08_sepal-user_sample_3_ALL_chip_0.tif",
            "value": None,
            "id": "271_RSP_H42",
            "date": datetime(2021, 7, 8).date(),
            "coordinate": (133.054459, -2.909185),
        },
        {
            "image": "/home/sepal-user/soil_moisture/papua_island/close_SMCmap_2022_04_17_DESC_sepal-user_phu_smm_stat_20230830_group_14.tif",
            "value": None,
            "id": "271_RSP_H42",
            "date": datetime(2022, 4, 17).date(),
            "coordinate": (133.054459, -2.909185),
        },
    ]

    # Run your function to get the results
    actual_results = find_pixel_values_for_coordinates(index, test_coords)

    # subset the results to only the ones that match with the expected_results images
    actual_results = [
        result
        for result in actual_results
        if result["image"] in [r["image"] for r in expected_results]
    ]

    # Sort results for easier comparison (optional)
    expected_results.sort(key=lambda x: x["image"])
    actual_results.sort(key=lambda x: x["image"])

    # Compare actual and expected
    assert (
        actual_results == expected_results
    ), f"Expected {expected_results}, got {actual_results}"

    print("Test done correctly.")

    return actual_results


test_find_pixel_values_for_coordinates(index)
