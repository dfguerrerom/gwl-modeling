from .directories import *
import ee
import os
import json
import toml


# read toml file
config_file = Path("__file__").parent / "gee_scripts/config.toml"


def get_project_from_conf():
    # Try to read the project name from the config file
    try:
        config = toml.load(config_file)
        return config.get("config").get("ee-project")
    except Exception as e:
        print(
            f"Error reading the config file, you should rename the file to config.toml"
        )


# This is a hard-copy of the original function from sepal_ui
def init_ee() -> None:
    r"""Initialize earth engine according using a token.

    THe environment used to run the tests need to have a EARTHENGINE_TOKEN variable.
    The content of this variable must be the copy of a personal credential file that you can find on your local computer if you already run the earth engine command line tool. See the usage question for a github action example.

    - Windows: ``C:\Users\USERNAME\\.config\\earthengine\\credentials``
    - Linux: ``/home/USERNAME/.config/earthengine/credentials``
    - MacOS: ``/Users/USERNAME/.config/earthengine/credentials``

    Note:
        As all init method of pytest-gee, this method will fallback to a regular ``ee.Initialize()`` if the environment variable is not found e.g. on your local computer.
    """
    if not ee.data._credentials:
        credential_folder_path = Path.home() / ".config" / "earthengine"
        credential_file_path = credential_folder_path / "credentials"

        if "EARTHENGINE_TOKEN" in os.environ and not credential_file_path.exists():

            # write the token to the appropriate folder
            ee_token = os.environ["EARTHENGINE_TOKEN"]
            credential_folder_path.mkdir(parents=True, exist_ok=True)
            credential_file_path.write_text(ee_token)

        # Extract the project name from credentials
        _credentials = json.loads(credential_file_path.read_text())

        project_id = (
            _credentials.get("project_id")
            or _credentials.get("project")
            or get_project_from_conf()
        )

        if not project_id:
            raise NameError(
                "The project name cannot be detected. "
                "Please set it using `earthengine set_project project_name`."
            )

        # Check if we are using a google service account
        if _credentials.get("type") == "service_account":
            ee_user = _credentials.get("client_email")
            credentials = ee.ServiceAccountCredentials(
                ee_user, str(credential_file_path)
            )
            ee.Initialize(credentials=credentials)
            ee.data._cloud_api_user_project = project_id
            return

        # if the user is in local development the authentication should
        # already be available
        ee.Initialize(project=project_id)

        print(f"Earth Engine initialized successfully, with {project_id}")


init_ee()
