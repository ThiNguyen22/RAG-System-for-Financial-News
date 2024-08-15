import pandas as pd
import influxdb_client
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tqdm.pandas()

def get_translate_text(text, max_length):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=max_length
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

token = "mytoken"
org = "myorg"
url = "https://database.bokai.online"
bucket = "VPS-Stock"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

read_api = client.query_api()

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", token='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', src_lang="vie_Latn"
)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", token='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')
model = model.to(device)


query = """
from(bucket: "VPS-Stock")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "news_stock")
  |> filter(fn: (r) => r["_field"] == "text" or r["_field"] == "title" or r["_field"] == "url" or r["_field"] == "last_update" or r["_field"] == "new_id")
  |> pivot(rowKey: ["_time", "Ma CK"], columnKey: ["_field"], valueColumn: "_value")
"""

df = read_api.query_data_frame(query).set_index("_time")
print(df)
df_exist = pd.read_pickle("data.pkl")

test = df.merge(df_exist[["title_en", "text_en"]], left_index=True, right_index=True, how="left")
test["last_update"] = pd.to_datetime(test["last_update"], format='mixed')

for idx, row in tqdm(test.iterrows(), total=len(test)):
    if isinstance(row["text"], str) and row["text"].strip() and (not row["text_en"]):
        row["title_en"] = get_translate_text(row["title"], max_length=150)
        row["text_en"] = get_translate_text(row["text"], max_length=300)
        print('title_en',row["title_en"])
        print('text_en',row["text_en"])

test.to_pickle("data.pkl")