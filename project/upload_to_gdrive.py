import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Load credential service account
creds = json.loads(os.environ["GDRIVE_CREDENTIALS"])
credentials = Credentials.from_service_account_info(
    creds, scopes=["https://www.googleapis.com/auth/drive"]
)

# Build Drive API
service = build("drive", "v3", credentials=credentials)

# Read Shared Drive Folder ID
SHARED_DRIVE_ID = os.environ["GDRIVE_FOLDER_ID"]


def upload_dir(local_dir_path, parent_drive_id):
    for item_name in os.listdir(local_dir_path):
        item_path = os.path.join(local_dir_path, item_name)

        # Folder
        if os.path.isdir(item_path):
            folder_meta = {
                "name": item_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_drive_id],
            }
            created_folder = (
                service.files()
                .create(body=folder_meta, fields="id", supportsAllDrives=True)
                .execute()
            )

            new_folder_id = created_folder["id"]
            print(f"Created folder: {item_name} (ID: {new_folder_id})")

            upload_dir(item_path, new_folder_id)

        # File
        else:
            print(f"Uploading file: {item_name}")
            file_meta = {"name": item_name, "parents": [parent_drive_id]}
            media = MediaFileUpload(item_path, resumable=True)
            service.files().create(
                body=file_meta, media_body=media, fields="id", supportsAllDrives=True
            ).execute()


# Upload folder mlruns/0
local_mlruns_0 = "./mlruns/0"

if os.path.exists(local_mlruns_0):
    upload_dir(local_mlruns_0, SHARED_DRIVE_ID)

print("======= Upload to Google Drive completed =======")
