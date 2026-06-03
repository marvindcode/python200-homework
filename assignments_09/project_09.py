import requests
import json
from datetime import date
from pathlib import Path

import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential


ACCOUNT_URL = "https://marvinctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "weather_raw.json"

    
url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=35.2271&longitude=-80.8431"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
)


response = requests.get(url)
response.raise_for_status()

data = response.json()
print("Extract complete.")


payload = json.dumps(data).encode("utf-8")
print(f"Serialized {len(payload)} bytes.")


today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

credential = DefaultAzureCredential()

container = ContainerClient(
    account_url=ACCOUNT_URL,
    container_name=CONTAINER,
    credential=credential
)

container.upload_blob(
    name=blob_path,data=payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {blob_path}")

    
print("\nBlobs in container:")
for blob in container.list_blobs():
    print(f"  {blob.name}  ({blob.size} bytes)")

raw = container.download_blob(blob_path).readall()

df = pd.DataFrame(json.loads(raw.decode("utf-8"))["hourly"])

print(f"\nFirst 5 rows:")
print(df.head())


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "wb") as file:
    file.write(raw)

print(f"\nSaved downloaded JSON to {OUTPUT_FILE}")