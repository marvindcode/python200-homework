#Video link: https://youtu.be/kKrJViepsbw

#Comment: Classifing weather ocnditions for outdoor runnning was not good use of LLM, because the output is expected to be one of three labels, and the logic can be easily implemented with deterministic code. With code can be with structured with conditions.  The deterministic code would be more consistent and accurate which makes it more efficient and less prone to errors compared to an LLM for this project.  


import json
import os
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

ACCOUNT_URL = "https://marvinctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

VALID_LABELS = {"good", "marginal", "bad"}

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )


today = "2026-06-02"

credential = DefaultAzureCredential()
container = ContainerClient(ACCOUNT_URL, CONTAINER, credential=credential)

raw = container.download_blob(f"raw/{today}/weather.json").readall()
data = json.loads(raw.decode("utf-8"))

hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    records.append({
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    })

print(f"Loaded {len(records)} records")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []
for i, record in enumerate(records[:24]):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )

    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 6 == 0:
        print(f"  Processed {i + 1} records...")

processed_path = f"processed/{today}/weather_classified.json"
container.upload_blob(processed_path, json.dumps(enriched).encode("utf-8"), overwrite=True)
print(f"Uploaded to {processed_path}")

df = pd.DataFrame(enriched)
print("\nLabel distribution:")
print(df["conditions"].value_counts())


print("\nFirst 5 rows:")
print(df.head())

os.makedirs("outputs", exist_ok=True)

with open("outputs/first_10_records.json", "w", encoding="utf-8") as file:
    json.dump(enriched[:10], file, indent=2)

print("\nSaved first 10 records to outputs/first_10_records.json")
