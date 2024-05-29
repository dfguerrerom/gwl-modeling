from pathlib import Path
import ee


explanatory_path = Path("data/11_explanatory_composites")
output_estimation_path = Path("data/12_estimated_gwl")
model_path = Path("data/10_models/")


# Create them if they don't exist
explanatory_path.mkdir(exist_ok=True)
output_estimation_path.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)


def get_export_folder(output_folder: str, base_folder: str = "gwl-modeling") -> Path:
    """Create a folder for the recipe in GEE"""

    def create_folder(folder: Path) -> None:
        try:
            if not ee.data.getInfo(str(folder)):
                ee.data.createAsset({"type": "FOLDER"}, str(folder))
        except Exception as e:
            return False

    project_folder = Path(f"projects/{ee.data._cloud_api_user_project}/assets/")
    base_path = project_folder / base_folder
    output_folder = Path(output_folder)
    full_path = base_path / output_folder

    # first check if the output folder exists
    if ee.data.getInfo(str(base_path / output_folder)):
        return full_path

    create_folder(base_path)

    # Check how many levels has the output_folder
    n_levels = len(Path(output_folder).parts)

    # incrementally create each of the levels in the path
    for i in range(1, n_levels + 1):
        new_folder = base_path / Path(*output_folder.parts[:i])
        create_folder(new_folder)

    return full_path
