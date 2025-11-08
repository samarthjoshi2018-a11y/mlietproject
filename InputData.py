from datetime import datetime
from pydantic import BaseModel


class Inputdata (BaseModel):
    user_id: str
    timestamp: datetime
    temperature: int
    time_of_the_day: str
    day_of_week:int
    weather_conditions:str
    frequency_crossing: int