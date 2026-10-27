import pandas as pd

INPUT = "rentals_enriched.jsonl"
OUTPUT = "rentals_clean.jsonl"

df = pd.read_json(INPUT, lines=True)
df = df.dropna(subset=["url"])
df = df.drop_duplicates(subset=["url"], keep="first").copy()

print("Content length: ", len(df))
print("Missing values: \n", df.isna().sum().sort_values(ascending=False).head(20))

numeric_columns = [
    "price",
    "rent",
    "area_m2",
    "rooms",
    "lat",
    "lng",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[(df["transaction"] == "RENT") & (df["type"] == "FLAT")
    & (df["currency"] == "PLN")].copy()

df = df.dropna(subset=["price", "area_m2"])
df = df[df["price"].between(500, 30000) & df["area_m2"].between(10, 300)].copy()

df["city_clean"] = (
    df["source_location"]
    .astype("string")
    .str.split(",")
    .str[0]
    .str.strip()
)

invalid_geo = (
    (df["lat"].notna() & df["lng"].notna())
    & (
        ~df["lat"].between(49.0, 55.1)
        | ~df["lng"].between(14.0, 24.2)
    )
)

print("\nInvalid geographic coordinates:", invalid_geo.sum())
df.loc[invalid_geo, ["lat", "lng"]] = float("nan")
incomplete_geo = df["lat"].isna() ^ df["lng"].isna()
df.loc[incomplete_geo, ["lat", "lng"]] = float("nan")
print("Clean records:", len(df))

print("Missing values after cleaning:")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("Apartments by city:")
print(df["city_clean"].value_counts())

df.to_json(OUTPUT, orient="records", force_ascii=False, indent=2)

columns = [
    "price",
    "area_m2",
    "rooms",
    "floor",
    "city_clean",
    "district",
    "lat",
    "lng",
]

print("Count of NaN columns: ", df[columns].isna().sum())

print("Apartments with coordinates: ", df[["lat", "lng"]].notna().all(axis=1).sum())