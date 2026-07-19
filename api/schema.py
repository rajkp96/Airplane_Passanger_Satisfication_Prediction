from pydantic import BaseModel

class Passenger(BaseModel):
    Age: int
    Gender: str
    Customer_Type: str
    Type_of_Travel: str
    Class: str

    Flight_Distance: int
    Departure_Delay: int

    Departure_and_Arrival_Time_Convenience: int
    Ease_of_Online_Booking: int
    Check_in_Service: int
    Online_Boarding: int
    Gate_Location: int
    On_board_Service: int
    Seat_Comfort: int
    Leg_Room_Service: int
    Cleanliness: int
    Food_and_Drink: int
    In_flight_Service: int
    In_flight_Wifi_Service: int
    In_flight_Entertainment: int
    Baggage_Handling: int

