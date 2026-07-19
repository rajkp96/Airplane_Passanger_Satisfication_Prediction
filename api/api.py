from fastapi import FastAPI
from schema import Passenger
import joblib
import pandas as pd
# Create FastAPI App

app = FastAPI(
    title="Airline Passenger Satisfaction Prediction API",
    description="Predict whether a passenger is Satisfied or Neutral/Dissatisfied",
    version="1.0"
)

# Load Model

model = joblib.load("C:/Users/Raj Kumar Patra/OneDrive/Documents/VS_code and Antigravity/Airplane_Passenger_Satisfication/model/Airline Passenger Satisfaction Prediction_model.pkl")

# Home Route

@app.get("/")
def home():
    return {"message": "Airline Passenger Satisfaction API is Running"}

# Prediction Route

@app.post("/predict")
def predict(data: Passenger):

    # Encoding

    class_map = {"Eco": 0,"Eco Plus": 1,"Business": 2}
    gender = 1 if data.Gender == "Male" else 0
    customer = 1 if data.Customer_Type == "Returning" else 0
    travel = 1 if data.Type_of_Travel == "Personal Travel" else 0
    class_type = class_map[data.Class]

    # Create DataFrame

    input_df = pd.DataFrame({

        "Age":[data.Age],
        "Class":[class_type],
        "Flight Distance":[data.Flight_Distance],
        "Departure Delay":[data.Departure_Delay],
        "Departure and Arrival Time Convenience":[data.Departure_and_Arrival_Time_Convenience],
        "Ease of Online Booking":[data.Ease_of_Online_Booking],
        "Check-in Service":[data.Check_in_Service],
        "Online Boarding":[data.Online_Boarding],
        "Gate Location":[data.Gate_Location],
        "On-board Service":[data.On_board_Service],
        "Seat Comfort":[data.Seat_Comfort],
        "Leg Room Service":[data.Leg_Room_Service],
        "Cleanliness":[data.Cleanliness],
        "Food and Drink":[data.Food_and_Drink],
        "In-flight Service":[data.In_flight_Service],
        "In-flight Wifi Service":[data.In_flight_Wifi_Service],
        "In-flight Entertainment":[data.In_flight_Entertainment],
        "Baggage Handling":[data.Baggage_Handling],
        "Gender_Male":[gender],
        "Customer Type_Returning":[customer],
        "Type of Travel_Personal":[travel]
    })

    # Prediction

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Return Result
    if prediction == 1:
        result = "Satisfied"
    else:
        result = "Neutral / Dissatisfied"

    return {

        "Prediction": result,
        "Prediction_Code": int(prediction),
        "Probability": round(float(probability),4)
    }