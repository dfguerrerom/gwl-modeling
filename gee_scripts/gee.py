from .parameters import explain_vars
from gee_scripts.directories import get_export_folder

import ee.batch


def export_classifier(training_data: ee.FeatureCollection, model_name: str):
    """Export and save a classifier to GEE.

    Args:
        training_data: table with all the explanatory variables and the response variable (gwl_cm)
        model_name: name of the model to export, this name refers to the    column that indicates if the sample belongs to the model or not.
    """

    features = training_data.filter(ee.Filter.eq(model_name, 1))
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

    task = ee.batch.Export.classifier.toAsset(classifier, description, model_gee_id)
    task.start()

    print(f"Exported model {model_gee_id}")
