# Video link: PASTE_YOUR_VIDEO_LINK_HERE
# https://youtu.be/xBlOLtTYBsg

import json
import os
from datetime import date

import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

load_dotenv()

CONTAINER = "pipeline-data"
DAY_RECORDS = 24  

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation"
        f"&forecast_days=7"
    )

    response = requests.get(url)
    response.raise_for_status()

    print(f"Extracted forecast data for ({latitude}, {longitude})")
    return response.json()


@task
def transform(data: dict, day_records: int) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    hourly = data["hourly"]

    records = []
    for i in range(len(hourly["time"])):
        records.append({
            "time": hourly["time"][i],
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        })

    for i, record in enumerate(records[:day_records]):
        user_msg = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )

        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"

        records[i]["classification"] = label

        if (i + 1) % 6 == 0:
            print(f"  Classified {i + 1}/{len(records)} records")

    print(f"Transform complete: {len(records)} records enriched")
    return records

@task
def load(enriched_records, blob_path: str) -> str:
    json_data = json.dumps(enriched_records, indent=2)
    byte_count = len(json_data.encode("utf-8"))

    blob_service_client = BlobServiceClient(
        account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
        credential=DefaultAzureCredential(),
    )

    blob_client = blob_service_client.get_blob_client(
        container="pipeline-data",
        blob=blob_path,
    )

    blob_client.upload_blob(json_data, overwrite=True)

    print(f"Uploaded to {blob_path} ({byte_count} bytes)")
    return blob_path


@flow(log_prints=True)
def etl_pipeline(
    latitude: float = 35.2271,
    longitude: float = -80.8431
):
    today = date.today().isoformat()

    blob_path = f"final/{today}/weather_etl.json"

    data = extract(latitude, longitude)

    enriched = transform(
        data,
        day_records=DAY_RECORDS
    )

    final_blob_path = load(enriched, blob_path)

    print(f"Full ETL Pipeline complete. Results at {final_blob_path}")

if __name__ == "__main__":
    etl_pipeline()