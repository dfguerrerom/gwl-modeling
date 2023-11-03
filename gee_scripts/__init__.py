from pathlib import Path


explanatory_path = Path("data/11_explanatory_composites")
output_estimation_path = Path("data/12_estimated_gwl")
model_path = Path("data/10_models/")


# Create them if they don't exist
explanatory_path.mkdir(exist_ok=True)
output_estimation_path.mkdir(exist_ok=True)
model_path.mkdir(exist_ok=True)
