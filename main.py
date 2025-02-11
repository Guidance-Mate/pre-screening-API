from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests, csv, json, random
from mangum import Mangum
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    "Severely - it bothered me a lot": 3
}


# -----------------------------------------------------------------------------
# Create synthetic training data and train a Random Forest model for each tool.
# In production, you would load a pre-trained model rather than training on random data.
# -----------------------------------------------------------------------------

# --- PHQ9 Model ---
# Assume 9 responses; labels are based on total score thresholds.
def generate_phq9_training_data(num_samples=100):
    X = np.random.randint(0, 4, (num_samples, 9))
    y = []
    for sample in X:
        total = sample.sum()
        if total <= 4:
            y.append("Minimal or none (0-4)")
        elif total <= 9:
            y.append("Mild (5-9)")
        elif total <= 14:
            y.append("Moderate (10-14)")
        elif total <= 19:
            y.append("Moderately severe (15-19)")
        else:
            y.append("Severe (20-27)")
    return X, np.array(y)


phq9_X, phq9_y = generate_phq9_training_data()
phq9_rf = RandomForestClassifier()
phq9_rf.fit(phq9_X, phq9_y)


# --- BAI Model ---
# Assume 21 responses; labels are based on sum thresholds.
def generate_bai_training_data(num_samples=100):
    X = np.random.randint(0, 4, (num_samples, 21))
    y = []
    for sample in X:
        total = sample.sum()
        if total <= 21:
            y.append("Low Anxiety (0-21)")
        elif total <= 35:
            y.append("Moderate Anxiety (22-35)")
        else:
            y.append("Severe Anxiety (36+)")
    return X, np.array(y)


bai_X, bai_y = generate_bai_training_data()
bai_rf = RandomForestClassifier()
bai_rf.fit(bai_X, bai_y)


# --- ASQ Model ---
# For ASQ, we create a simple feature vector:
#   Feature 1: 1 if "None of the above" is in the selected options; else 0.
#   Feature 2: 1 if the acuity response equals "Yes" (case-insensitive); else 0.
#   Feature 3: Length of the "how_and_when" text.
#   Feature 4: Length of the "please_describe" text.
def generate_asq_training_data(num_samples=100):
    X = []
    y = []
    for _ in range(num_samples):
        # Randomly decide whether the client selected "None of the above"
        none_option = random.choice([0, 1])
        # Randomly decide whether the acuity response is "Yes"
        yes_acuity = random.choice([0, 1])
        # Random text lengths (could be normalized)
        how_and_when_len = random.randint(0, 100)
        please_describe_len = random.randint(0, 200)
        X.append([none_option, yes_acuity, how_and_when_len, please_describe_len])
        # Use a rule similar to the original:
        # If "None of the above" was selected then "No Risk"
        # Else if acuity is "Yes" then "Acute Positive Screen"
        # Else "Non-Acute Positive Screen"
        if none_option == 1:
            y.append("No Risk")
        elif yes_acuity == 1:
            y.append("Acute Positive Screen")
        else:
            y.append("Non-Acute Positive Screen")
    return np.array(X), np.array(y)


asq_X, asq_y = generate_asq_training_data()
asq_rf = RandomForestClassifier()
asq_rf.fit(asq_X, asq_y)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Combined Mental Health Tool API is running."}


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "message": "API is running and accessible."}


# --- PHQ9 Analysis Endpoint ---
@app.get("/analyze/phq9")
def analyze_phq9(first_name: str, last_name: str, middle_name: str = "", suffix: str = ""):
    input_name = f"{first_name} {middle_name} {last_name} {suffix}".strip()
    try:
        response = requests.get(PHQ9_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        header = next(reader)

        for row in reader:
            # Assume the last four columns are name parts
            row_name = f"{row[-4]} {row[-3]} {row[-2]} {row[-1]}".strip()
            if row_name.lower() == input_name.lower():
                # Extract responses from columns 1 to -4 (expecting 9 responses)
                responses = row[1:-4]
                if len(responses) != 9:
                    raise HTTPException(status_code=400, detail="Unexpected number of responses for PHQ9.")
                features = [phq9_response_mapping.get(r.strip(), 0) for r in responses]
                features_array = np.array(features).reshape(1, -1)
                prediction = phq9_rf.predict(features_array)[0]
                total_score = sum(features)
                if prediction in ["Minimal or none (0-4)", "Mild (5-9)"]:
                    primary_impression = "The client may have mild or no mental health concerns."
                    additional_impressions = []
                    tool_recommendations = []
                else:
                    primary_impression = "The client might be experiencing more significant mental health concerns."
                    additional_impressions = [
                        "The analysis suggests potential Depression.",
                        "Physical symptoms may be affecting the client.",
                        "Overall well-being might require attention."
                    ]
                    tool_recommendations = [
                        "Tools for Depression",
                        "Tools for Physical Symptoms",
                        "Tools for Well-Being"
                    ]
                return {
                    "client_name": input_name.title(),
                    "total_score": total_score,
                    "interpretation": prediction,
                    "primary_impression": primary_impression,
                    "additional_impressions": additional_impressions,
                    "tool_recommendations": tool_recommendations
                }
        raise HTTPException(status_code=404, detail=f"Client '{input_name}' not found in PHQ9 data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PHQ9 data: {e}")


# --- BAI Analysis Endpoint ---
@app.get("/analyze/bai")
def analyze_bai(first_name: str, last_name: str, middle_name: str = "", suffix: str = ""):
    input_name = f"{first_name} {middle_name} {last_name} {suffix}".strip()
    try:
        response = requests.get(BAI_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        header = next(reader)

        for row in reader:
            row_name = f"{row[-4]} {row[-3]} {row[-2]} {row[-1]}".strip()
            if row_name.lower() == input_name.lower():
                responses = row[1:-4]
                # Expecting 21 responses for BAI
                if len(responses) != 21:
                    raise HTTPException(status_code=400, detail="Unexpected number of responses for BAI.")
                features = [bai_response_mapping.get(r.strip(), 0) for r in responses]
                features_array = np.array(features).reshape(1, -1)
                prediction = bai_rf.predict(features_array)[0]
                total_score = sum(features)
                if prediction == "Low Anxiety (0-21)":
                    primary_impression = "The client may have mild or no anxiety concerns."
                    additional_impressions = []
                    tool_recommendations = []
                else:
                    primary_impression = "The client might be experiencing anxiety or related concerns."
                    additional_impressions = [
                        "Further evaluation may be needed for Anxiety symptoms.",
                        "Symptoms of Trauma or PTSD may be present.",
                        "Youth Mental Health factors may require attention."
                    ]
                    tool_recommendations = [
                        "Tools for Anxiety",
                        "Tools for Trauma & PTSD",
                        "Tools for Youth Mental Health"
                    ]
                return {
                    "client_name": input_name.title(),
                    "total_score": total_score,
                    "interpretation": prediction,
                    "primary_impression": primary_impression,
                    "additional_impressions": additional_impressions,
                    "tool_recommendations": tool_recommendations
                }
        raise HTTPException(status_code=404, detail=f"Client '{input_name}' not found in BAI data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing BAI data: {e}")


# --- ASQ Analysis Endpoint ---
@app.get("/analyze/asq")
def analyze_asq(first_name: str, last_name: str, middle_name: str = "", suffix: str = ""):
    input_name = f"{first_name} {middle_name} {last_name} {suffix}".strip()
    try:
        response = requests.get(ASQ_URL)
        response.raise_for_status()
        data = response.text.splitlines()
        reader = csv.reader(data)
        header = next(reader)

        for row in reader:
            row_name = f"{row[-4]} {row[-3]} {row[-2]} {row[-1]}".strip()
            if row_name.lower() == input_name.lower():
                # Extract ASQ-specific responses:
                #   selected_options (column index 2), how_and_when (index 3),
                #   acuity_response (index 5), please_describe (index 6)
                selected_options_raw = row[2].strip()
                how_and_when = row[3].strip()
                acuity_response = row[5].strip()
                please_describe = row[6].strip()
                selected_options = [opt.strip() for opt in selected_options_raw.split(",")]

                # Feature extraction:
                none_option = 1 if "None of the above" in selected_options else 0
                yes_acuity = 1 if acuity_response.lower() == "yes" else 0
                how_and_when_len = len(how_and_when)
                please_describe_len = len(please_describe)
                features = [none_option, yes_acuity, how_and_when_len, please_describe_len]
                features_array = np.array(features).reshape(1, -1)
                prediction = asq_rf.predict(features_array)[0]

                if prediction == "No Risk":
                    primary_impression = "The client has no risk of suicidal thoughts or behaviors."
                    additional_impressions = []
                    suggested_tools = []
                elif prediction == "Acute Positive Screen":
                    primary_impression = "The client is at imminent risk of suicide and requires immediate safety and mental health evaluation."
                    additional_impressions = ["The client requires a STAT safety/full mental health evaluation."]
                    suggested_tools = ["Tools for Suicide", "Immediate Mental Health Safety Plan"]
                else:
                    primary_impression = "The client is at potential risk of suicide and requires a brief suicide safety assessment."
                    additional_impressions = ["The client requires a brief suicide safety assessment."]
                    suggested_tools = ["Tools for Suicide", "Suicide Risk Assessment Tools"]

                return {
                    "client_name": input_name.title(),
                    "selected_options": selected_options,
                    "how_and_when": how_and_when or "N/A",
                    "please_describe": please_describe or "N/A",
                    "acuity_response": acuity_response or "N/A",
                    "interpretation": prediction,
                    "primary_impression": primary_impression,
                    "additional_impressions": additional_impressions,
                    "suggested_tools": suggested_tools
                }
        raise HTTPException(status_code=404, detail=f"Client '{input_name}' not found in ASQ data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ASQ data: {e}")


handler = Mangum(app)
