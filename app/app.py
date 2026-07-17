import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("C:/Users/Raj Kumar Patra/OneDrive/Documents/VS_code and Antigravity/Airplane_Passenger_Satisfication/model/Airline Passenger Satisfaction Prediction_model.pkl")
scaler = joblib.load("C:/Users/Raj Kumar Patra/OneDrive/Documents/VS_code and Antigravity/Airplane_Passenger_Satisfication/model/final_scaler.pkl")


st.set_page_config(page_title="Airline Passenger Satisfaction",page_icon="✈️",layout="wide")
st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write("Predict whether a passenger is Satisfied or Neutral/Dissatisfied")
st.sidebar.header("Passenger Details")

#User Inputs
age = st.sidebar.slider("Age",18,80,35)
flight_distance = st.sidebar.number_input("Flight Distance",50,5000,1000)
departure_delay = st.sidebar.number_input("Departure Delay",0,180,0)
class_type = st.sidebar.selectbox("Class",["Economy","Economy Plus","Business"])
gender = st.sidebar.selectbox("Gender",["Female","Male"])
customer_type = st.sidebar.selectbox("Customer Type",["First-time","Returning"])
travel_type = st.sidebar.selectbox("Type of Travel",["Business","Personal"])

#Service Ratings
departure_arrival = st.sidebar.slider("Departure and Arrival Time Convenience", 0, 5, 3)
online_booking = st.sidebar.slider("Ease of Online Booking", 0, 5, 3)
checkin = st.sidebar.slider("Check-in Service", 0, 5, 3)
online_boarding = st.sidebar.slider("Online Boarding", 0, 5, 3)
gate = st.sidebar.slider( "Gate Location", 0, 5, 3)
onboard = st.sidebar.slider("On-board Service", 0, 5, 3)
seat = st.sidebar.slider("Seat Comfort", 0, 5, 3)
legroom = st.sidebar.slider("Leg Room Service", 0, 5, 3)
cleanliness = st.sidebar.slider("Cleanliness", 0, 5, 3)
food = st.sidebar.slider( "Food and Drink", 0, 5, 3)
inflight_service = st.sidebar.slider("In-flight Service", 0, 5, 3)
wifi = st.sidebar.slider("In-flight Wifi Service", 0, 5, 3)
entertainment = st.sidebar.slider("In-flight Entertainment", 0, 5, 3)
baggage = st.sidebar.slider("Baggage Handling", 0, 5, 3)


#Convert Inputs
class_map={"Economy":0,"Economy Plus":1,"Business":2}
gender=int(gender=="Male")
customer=int(customer_type=="Returning")
travel=int(travel_type=="Personal")

#Input Dataframe
input_df = pd.DataFrame({
    "Age":[age],
    "Class":[class_map[class_type]],
    "Flight Distance":[flight_distance],
    "Departure Delay":[departure_delay],


    "Departure and Arrival Time Convenience":[departure_arrival],
    "Ease of Online Booking":[online_booking],
    "Check-in Service":[checkin],
    "Online Boarding":[online_boarding],
    "Gate Location":[gate],
    "On-board Service":[onboard],
    "Seat Comfort":[seat],
    "Leg Room Service":[legroom],
    "Cleanliness":[cleanliness],
    "Food and Drink":[food],
    "In-flight Service":[inflight_service],
    "In-flight Wifi Service":[wifi],
    "In-flight Entertainment":[entertainment],
    "Baggage Handling":[baggage],

    "Gender_Male":[gender],
    "Customer Type_Returning":[customer],
    "Type of Travel_Personal":[travel]})


if st.button("Predict Satisfaction"):

    prediction=model.predict(input_df)[0]
    probability=model.predict_proba(input_df)[0][1]
    if prediction==1:
        st.success("Passenger is Satisfied 😊")
    else:
        st.error("Passenger is Neutral / Dissatisfied 😞")

    st.metric("Probability",f"{probability*100:.2f}%")