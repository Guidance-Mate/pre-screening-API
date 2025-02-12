from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import csv
import json
import random
from mangum import Mangum

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# URLs for assessment data
PHQ9_URL = "https://docs.google.com/spreadsheets/d/1D312sgbt_nOsT668iaUrccAzQ3oByUT0peXS8LYL5wg/export?format=csv"
ASQ_URL = "https://docs.google.com/spreadsheets/d/1TiU8sv5cJg30ZL3fqPSmBwJJbB7h2xv1NNbKo4ZIydU/export?format=csv"
BAI_URL = "https://docs.google.com/spreadsheets/d/1f7kaFuhCv6S_eX4EuIrlhZFDR7W5MhQpJSXHznlpJEk/export?format=csv"

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

                primary_impression = (
                    "The client may have mild or no mental health concerns."
                    if interpretation in ["Minimal or none (0-4)", "Mild (5-9)"]
                    else "The client might be experiencing more significant mental health concerns."
                )

                additional_impressions = [
                    "The analysis suggests the client might be experiencing Depression.",
                    "Physical symptoms may be affecting the client.",
                    "The client's overall well-being might require attention."
                ] if interpretation not in ["Minimal or none (0-4)", "Mild (5-9)"] else []

                tool_recommendations = [
                    "Tools for Depression",
                    "Tools for Physical Symptoms",
                    "Tools for Well-Being"
                ] if interpretation not in ["Minimal or none (0-4)", "Mild (5-9)"] else []

                results["phq9"] = {
                    "client_name": input_name.title(),
                    "total_score": total_score,
                    "interpretation": interpretation,
                    "primary_impression": primary_impression,
                    "additional_impressions": additional_impressions,
                    "tool_recommendations": tool_recommendations
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

                interpretation = "No Risk"
                primary_impression = "The client has no risk of suicidal thoughts or behaviors."
                additional_impressions = []
                suggested_tools = []

                if "Yes" in acuity_response:
                    interpretation = "Acute Positive Screen"
                    primary_impression = "The client is at imminent risk of suicide and requires immediate safety and mental health evaluation."
                    additional_impressions = ["The client requires a STAT safety/full mental health evaluation."]
                    suggested_tools = ["Tools for Suicide", "Immediate Mental Health Safety Plan"]

                results["asq"] = {
                    "client_name": input_name.title(),
                    "selected_options": selected_options,
                    "acuity_response": acuity_response,
                    "interpretation": interpretation,
                    "primary_impression": primary_impression,
                    "additional_impressions": additional_impressions,
                    "suggested_tools": suggested_tools
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

                primary_impression = "The client may have mild or no anxiety concerns." if interpretation == "Low Anxiety (0-21)" else "The client might be experiencing anxiety or related concerns."
                additional_impressions = [
                    "Further evaluation may be needed for Anxiety symptoms.",
                    "Symptoms of Trauma or PTSD were noted.",
                    "Youth Mental Health factors may require attention."
                ] if interpretation != "Low Anxiety (0-21)" else []

                tool_recommendations = [
                    "Tools for Anxiety",
                    "Tools for Trauma & PTSD",
                    "Tools for Youth Mental Health"
                ] if interpretation != "Low Anxiety (0-21)" else []

                results["bai"] = {
                    "client_name": input_name.title(),
                    "total_score": total_score,
                    "interpretation": interpretation,
                    "primary_impression": primary_impression,
                    "additional_impressions": additional_impressions,
                    "tool_recommendations": tool_recommendations
                }
                break

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

handler = Mangum(app)
