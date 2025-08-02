from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from fuzzywuzzy import process
import pandas as pd

# Load the dataset
file = "reviews.pkl"
with open(file, "rb") as fileob:
    dataset = pickle.load(fileob)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class String(BaseModel):
    name: str

@app.post('/hotel_name/')
async def ratings(name: String):
    name = name.name.strip()  # Remove leading/trailing whitespace
    hotel_names = dataset['Hotel_name'].str.lower().unique()  # Case-insensitive check
    if name.lower() not in hotel_names:
        similar_hotels = process.extract(name, dataset['Hotel_name'].unique(), limit=5)
        similar_hotels_names = [h[0] for h in similar_hotels]

        sim = {index + 1: name for index, name in enumerate(similar_hotels_names)}
        

        return sim
    else:
        result = dataset[dataset['Hotel_name'].str.lower() == name.lower()].round(2).to_dict(orient='records')
        return result
