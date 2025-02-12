from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests, csv
from mangum import Mangum

app = FastAPI()

# Allow all origins (use a specific list for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URLs for CSV data
PHQ9_URL = "https://docs.google.com/spreadsheets/d/1D312sgbt_nOsT668iaUrccAzQ3oByUT0peXS8LYL5wg/export?format=csv"
BAI_URL = "https://docs.google.com/spreadsheets/d/1f7kaFuhCv6S_eX4EuIrlhZFDR7W5MhQpJSXHznlpJEk/export?format=csv"
ASQ_URL = "https://docs.google.com/spreadsheets/d/1TiU8sv5cJg30ZL3fqPSmBwJJbB7h2xv1NNbKo4ZIydU/export?format=csv"

# Response mappings for PHQ9 and BAI
phq9_response_mapping = {
    "Not at all": 0,
    "Several Days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}

bai_response_mapping = {
    "Not at all": 0,
    "Mildly, but it didn't bother me much": 1,
    "Moderately - it wasn't pleasant at times": 2,
    "Severely - it bothered me a lot": 3,
}

@app.get("/")
def root():
    return {"message": "Combined Mental Health Tool API is running."}

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "message": "API is running and accessible."}

def fetch_and_process_csv(url, input_name, expected_responses, response_mapping=None):
    """Fetches the data from a CSV file and extracts user responses."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.text.splitlines()
        reader = csv.reader(data)
        header = next(reader)

        for row in reader:
            row_name = f"{row[-4]} {row[-3]} {row[-2]} {row[-1]}".strip().lower()
            if row_name == input_name.lower():
                responses = row[1:-4]
                if len(responses) != expected_responses:
                    raise HTTPException(status_code=400, detail=f"Unexpected number of responses for this tool.")

                if response_mapping:
                    scores = [response_mapping.get(r.strip(), 0) for r in responses]
                else:
                    scores = responses  # ASQ does not use mapping

                return scores

        raise HTTPException(status_code=404, detail=f"Client '{input_name}' not found in data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

def interpret_phq9(scores):
    """Interprets PHQ-9 scores based on total sum."""
    total = sum(scores)
    if total <= 4:
        return "Minimal or none (0-4)", "The client may have mild or no mental health concerns.", []
    elif total <= 9:
        return "Mild (5-9)", "The client may have mild depressive symptoms.", []
    elif total <= 14:
        return "Moderate (10-14)", "The client might be experiencing moderate depressive symptoms.", ["Counseling recommended."]
    elif total <= 19:
        return "Moderately severe (15-19)", "The client might be experiencing significant depressive symptoms.", ["Clinical assessment needed."]
    else:
        return "Severe (20-27)", "The client might have severe depressive symptoms and requires immediate attention.", ["Urgent intervention recommended."]

def interpret_bai(scores):
    """Interprets BAI scores based on total sum."""
    total = sum(scores)
    if total <= 21:
        return "Low Anxiety (0-21)", "The client has mild or no anxiety symptoms.", []
    elif total <= 35:
        return "Moderate Anxiety (22-35)", "The client might be experiencing moderate anxiety.", ["Relaxation techniques suggested."]
    else:
        return "Severe Anxiety (36+)", "The client may have severe anxiety symptoms requiring clinical intervention.", ["Therapy or medical evaluation recommended."]

def interpret_asq(selected_options, acuity_response):
    """Interprets ASQ responses."""
    if "None of the above" in selected_options:
        return "No Risk", "The client has no risk of suicidal thoughts or behaviors.", []
    elif acuity_response.lower() == "yes":
        return "Acute Positive Screen", "The client is at imminent risk of suicide and requires immediate evaluation.", ["Immediate safety plan required."]
    else:
        return "Non-Acute Positive Screen", "The client may require further assessment for suicide risk.", ["Suicide risk assessment tools recommended."]

@app.get("/analyze")
def analyze_all(first_name: str, last_name: str, middle_name: str = "", suffix: str = ""):
    input_name = f"{first_name} {middle_name} {last_name} {suffix}".strip()

    # Fetch results from CSVs
    phq9_scores = fetch_and_process_csv(PHQ9_URL, input_name, 9, phq9_response_mapping)
    bai_scores = fetch_and_process_csv(BAI_URL, input_name, 21, bai_response_mapping)
    asq_data = fetch_and_process_csv(ASQ_URL, input_name, 0)  # No response mapping for ASQ

    # Interpret results
    phq9_interpretation, phq9_primary, phq9_recommendations = interpret_phq9(phq9_scores)
    bai_interpretation, bai_primary, bai_recommendations = interpret_bai(bai_scores)

    # Extract ASQ-specific responses
    selected_options = asq_data[2].split(",") if asq_data and len(asq_data) > 2 else []
    acuity_response = asq_data[5] if asq_data and len(asq_data) > 5 else ""
    asq_interpretation, asq_primary, asq_recommendations = interpret_asq(selected_options, acuity_response)

    return {
        "phq9": {
            "total_score": sum(phq9_scores),
            "interpretation": phq9_interpretation,
            "primary_impression": phq9_primary,
            "additional_impressions": [],
            "tool_recommendations": phq9_recommendations,
        },
        "bai": {
            "total_score": sum(bai_scores),
            "interpretation": bai_interpretation,
            "primary_impression": bai_primary,
            "additional_impressions": [],
            "tool_recommendations": bai_recommendations,
        },
        "asq": {
            "selected_options": selected_options,
            "acuity_response": acuity_response,
            "interpretation": asq_interpretation,
            "primary_impression": asq_primary,
            "additional_impressions": [],
            "tool_recommendations": asq_recommendations,
        }
    }

handler = Mangum(app)
