from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import csv
import json
import random
from mangum import Mangum  # Required for AWS Lambda compatibility

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use specific origins in production for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

# URLs for assessment data
PHQ9_URL = "https://docs.google.com/spreadsheets/d/1D312sgbt_nOsT668iaUrccAzQ3oByUT0peXS8LYL5wg/export?format=csv"
ASQ_URL = "https://docs.google.com/spreadsheets/d/1TiU8sv5cJg30ZL3fqPSmBwJJbB7h2xv1NNbKo4ZIydU/export?format=csv"
BAI_URL = "https://docs.google.com/spreadsheets/d/1f7kaFuhCv6S_eX4EuIrlhZFDR7W5MhQpJSXHznlpJEk/export?format=csv"

# Load phrases
with open("phrases_phq9.json", "r") as f:
    phrases_phq9 = json.load(f)

with open("phrases_asq.json", "r") as f:
    phrases_asq = json.load(f)

with open("phrases_bai.json", "r") as f:
    phrases_bai = json.load(f)

# Response mappings
response_mapping_phq9 = {
    "Not at all": 0,
    "Several Days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

response_mapping_bai = {
    "Not at all": 0,
    "Mildly, but it didn't bother me much": 1,
    "Moderately - it wasn't pleasant at times": 2,
    "Severely - it bothered me a lot": 3
}

# Interpretation functions
def get_phq9_interpretation(score):
    if score <= 4:
        return "Minimal or none (0-4)"
    elif score <= 9:
        return "Mild (5-9)"
    elif score <= 14:
        return "Moderate (10-14)"
    elif score <= 19:
        return "Moderately severe (15-19)"
    else:
        return "Severe (20-27)"

def get_bai_interpretation(score):
    if score <= 21:
        return "Low Anxiety (0-21)"
    elif score <= 35:
        return "Moderate Anxiety (22-35)"
    else:
        return "Severe Anxiety (36+)"

@app.get("/")
def root():
    return {"message": "Mental Health Assessment API is running."}

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "message": "API is running and accessible."}

@app.get("/analyze")
def analyze_assessments(first_name: str, last_name: str, middle_name: str = "", suffix: str = ""):
    input_name = f"{first_name} {middle_name} {last_name} {suffix}".strip()
    
    try:
        results = {}

        # PHQ-9 Analysis
        response = requests.get(PHQ9_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        next(reader)  # Skip header

        for row in reader:
            row_name = f"{row[-4]} {row[-3]} {row[-2]} {row[-1]}".strip()
            if row_name.lower() == input_name.lower():
                responses = row[1:-4]
                total_score = sum(response_mapping_phq9.get(r.strip(), 0) for r in responses)
                interpretation = get_phq9_interpretation(total_score)
                results["PHQ-9"] = {
                    "total_score": total_score,
                    "interpretation": interpretation
                }
                break

        # ASQ Analysis
        response = requests.get(ASQ_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        next(reader)  # Skip header

        for row in reader:
            row_name = f"{row[-4]} {row[-3]} {row[-2]} {row[-1]}".strip()
            if row_name.lower() == input_name.lower():
                selected_options = [option.strip() for option in row[2].strip().split(",")]
                acuity_response = row[5].strip()

                if "None of the above" in selected_options:
                    interpretation = "No Risk"
                elif "Yes" in acuity_response:
                    interpretation = "Acute Positive Screen"
                else:
                    interpretation = "Non-Acute Positive Screen"

                results["ASQ"] = {
                    "selected_options": selected_options,
                    "acuity_response": acuity_response,
                    "interpretation": interpretation
                }
                break

        # BAI Analysis
        response = requests.get(BAI_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        next(reader)  # Skip header

        for row in reader:
            row_name = f"{row[-4]} {row[-3]} {row[-2]} {row[-1]}".strip()
            if row_name.lower() == input_name.lower():
                responses = row[1:-4]
                total_score = sum(response_mapping_bai.get(r.strip(), 0) for r in responses)
                interpretation = get_bai_interpretation(total_score)
                results["BAI"] = {
                    "total_score": total_score,
                    "interpretation": interpretation
                }
                break

        if not results:
            raise HTTPException(status_code=404, detail=f"Client '{input_name}' not found.")

        return {"client_name": input_name.title(), "assessments": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

# AWS Lambda handler
handler = Mangum(app)
