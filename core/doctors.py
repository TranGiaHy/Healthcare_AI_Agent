from typing import List, Dict
import pandas as pd
import folium
from .config import DOCTORS_PATH

def load_doctors() -> pd.DataFrame:
    return pd.read_csv(DOCTORS_PATH)

def find_doctors(specialty: str, city: str, top_k: int = 3) -> List[Dict]:
    df = load_doctors()
    filtered = df.copy()
    if specialty:
        filtered = filtered[filtered["specialty"].str.contains(specialty, case=False, na=False)]
    if city:
        filtered = filtered[filtered["city"].str.contains(city, case=False, na=False)]
    if filtered.empty and specialty:
        filtered = df[df["specialty"].str.contains(specialty, case=False, na=False)]
    if filtered.empty:
        filtered = df[df["specialty"].str.contains("Internal", case=False, na=False)]
    
    filtered = filtered.head(top_k)
    return [
        {
            "name": row["name"],
            "specialty": row["specialty"],
            "city": row["city"],
            "rating": float(row["rating"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        } for _, row in filtered.iterrows()
    ]

def create_doctor_map(doctors: List[Dict]) -> str:
    if not doctors:
        return "<p>No doctors found.</p>"
    first = doctors[0]
    m = folium.Map(location=[first["latitude"], first["longitude"]], zoom_start=12)
    for doc in doctors:
        folium.Marker(
            location=[doc["latitude"], doc["longitude"]],
            popup=f"{doc['name']} ({doc['specialty']}) - {doc['rating']}",
        ).add_to(m)
    return m._repr_html_()
