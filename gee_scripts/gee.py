from pathlib import Path
from gee_scripts.get_sources import get_explanatory_composite
from .parameters import explain_vars
from gee_scripts.directories import create_image_collection, get_export_folder

import ee.batch


def export_classifier(training_data: ee.FeatureCollection, model_name: str):
    """Export and save a classifier to GEE.

    Args:
        training_data: table with all the explanatory variables and the response variable (gwl_cm)
        model_name: name of the model to export, this name refers to the    column that indicates if the sample belongs to the model or not.
    """

    features = training_data
    print(f"Exporting model {model_name} with {features.size().getInfo()} samples")

    n_trees = 250

    classifier = (
        ee.Classifier.smileRandomForest(n_trees)
        .setOutputMode("REGRESSION")
        .train(features=features, classProperty="gwl_cm", inputProperties=explain_vars)
    )

    description = f"RandomForest_{model_name}_trees_{n_trees}"

    # Create a folder to store the models
    model_gee_id = str(get_export_folder("models") / description)

    if ee.data.getInfo(model_gee_id):
        raise ValueError(f"Model {model_gee_id} already exists")

    task = ee.batch.Export.classifier.toAsset(
        **{
            "classifier": classifier,
            "description": description,
            "assetId": model_gee_id,
        }
    )
    task.start()

    print(f"Exported model {model_gee_id}")

    return model_gee_id


def estimate_to_gee(
    aoi_name: str,
    ee_aoi: ee.Geometry,
    target_date: str,
    ee_classifier,
):
    """Export the estimated GWL image using a given model and target date.


    Args:
        target_date: must coincide with the date of S1 image
    """

    model_name = Path(ee_classifier.getInfo()["id"]).name

    # Get explanatory composite closest to target date
    image = get_explanatory_composite(
        target_date=target_date,
        ee_region=ee_aoi,
    ).select(explain_vars)

    output_image_name = f"{aoi_name}_{target_date}"
    estimated_image = (
        image.select(explain_vars)
        .classify(ee_classifier)
        .set({"model": model_name, "date": target_date})
    )

    export_folder = get_export_folder(output_folder=f"estimation/best_models/")
    image_collection_path = create_image_collection(export_folder / model_name)

    # create export task
    return ee.batch.Export.image.toAsset(
        **{
            "image": estimated_image,
            "description": output_image_name,
            "assetId": str(image_collection_path / output_image_name),
            "region": ee_aoi,
            "scale": 100,
        }
    )
