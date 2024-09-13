"""This script uploads files from a local folder to Google Drive using the Google Drive API."""

import asyncio
import os
import json
from pathlib import Path
from aiogoogle import Aiogoogle
from aiogoogle.auth.creds import UserCreds, ClientCreds
import toml

# Load credentials from config.toml
conf_file = Path(__file__).parent / "config.toml"
client_secrets = toml.load(conf_file)["config"]
print(client_secrets)


client_creds = ClientCreds(
    client_id=client_secrets["client_id"],
    client_secret=client_secrets["client_secret"],
    scopes=["https://www.googleapis.com/auth/drive"],
    redirect_uri=client_secrets["redirect_uri"],
)


# You will need to authorize and get user credentials
async def authorize_user():
    async with Aiogoogle(client_creds=client_creds) as aiogoogle:
        auth_url = aiogoogle.oauth2.authorization_url()
        print(f"Go to the following URL to authorize:\n{auth_url}\n")
        code = input("Enter the authorization code: ")
        user_creds = await aiogoogle.oauth2.build_user_creds(grant=code)
        print(f"User credentials: {user_creds}")

        # Save user credentials for future use
        with open("user_creds.json", "w") as f:
            json.dump(user_creds.as_dict(), f)
        return user_creds


async def upload_files():
    # Check if user credentials exist
    if not os.path.exists("user_creds.json"):
        user_creds = await authorize_user()
    else:
        with open("user_creds.json", "r") as f:
            user_creds_data = json.load(f)
            user_creds = UserCreds(**user_creds_data)

    async with Aiogoogle(user_creds=user_creds, client_creds=client_creds) as aiogoogle:
        drive = await aiogoogle.discover("drive", "v3")

        folder_path = "/home/dguerrero/1_modules/gwl-modeling/data/1_bosf_data"

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                metadata = {"name": filename}
                with open(file_path, "rb") as file_content:
                    response = await aiogoogle.as_user(
                        drive.files.create(
                            upload_file=file_content, fields="id", json=metadata
                        )
                    )
                print(f'Uploaded {filename} with file ID {response["id"]}')


if __name__ == "__main__":
    asyncio.run(upload_files())
