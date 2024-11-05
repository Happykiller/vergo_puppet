# app\apis\models\weather_model_data.py
from enum import IntEnum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

class WeatherCode(IntEnum):
    """
    Enum representing weather conditions by codes.
    
    Attributes:
        CLEAR (int): Clear sky.
        MAINLY_CLEAR (int): Mostly clear sky.
        PARTLY_CLOUDY (int): Partly cloudy.
        OVERCAST (int): Overcast sky.
        FOG (int): Fog.
        DRIZZLE (int): Drizzle.
        FREEZING_DRIZZLE (int): Freezing drizzle.
        RAIN (int): Rain.
        FREEZING_RAIN (int): Freezing rain.
        SNOW_FALL (int): Snowfall.
        SNOW_GRAINS (int): Snow grains.
        RAIN_SHOWER (int): Rain shower.
        SNOW_SHOWER (int): Snow shower.
        THUNDERSTORM (int): Thunderstorm.
        THUNDERSTORM_WITH_HAIL (int): Thunderstorm with hail.
        OTHER (int): Other condition.
        OTHER1-OTHER8 (int): Additional unspecified weather conditions.
    """
    
    CLEAR = 0                  # Clear sky
    MAINLY_CLEAR = 1           # Mostly clear
    PARTLY_CLOUDY = 2          # Partly cloudy
    OVERCAST = 3               # Overcast
    FOG = 4                    # Fog
    DRIZZLE = 5                # Drizzle
    FREEZING_DRIZZLE = 6       # Freezing drizzle
    RAIN = 7                   # Rain
    FREEZING_RAIN = 8          # Freezing rain
    SNOW_FALL = 9              # Snowfall
    SNOW_GRAINS = 10           # Snow grains
    RAIN_SHOWER = 11           # Rain shower
    SNOW_SHOWER = 12           # Snow shower
    THUNDERSTORM = 13          # Thunderstorm
    THUNDERSTORM_WITH_HAIL = 14 # Thunderstorm with hail
    OTHER = 15                 # Other condition
    OTHER1 = 16                # Additional condition
    OTHER2 = 17                # Additional condition
    OTHER3 = 18                # Additional condition
    OTHER4 = 19                # Additional condition
    OTHER5 = 20                # Additional condition
    OTHER6 = 21                # Additional condition
    OTHER7 = 22                # Additional condition
    OTHER8 = 23                # Additional condition

class WeatherModelData(BaseModel):
    """
    WeatherModelData represents weather observation data for use in various models.

    Attributes:
        time (datetime): Timestamp of the weather observation.
        temp (Optional[float]): Temperature in Celsius.
        dwpt (Optional[float]): Dew point in Celsius.
        rhum (Optional[float]): Relative humidity in percentage.
        prcp (Optional[float]): Precipitation in millimeters.
        snow (Optional[float]): Snowfall in millimeters.
        wdir (Optional[float]): Wind direction in degrees (0 to 360).
        wspd (Optional[float]): Wind speed in km/h.
        wpgt (Optional[float]): Maximum wind gust in km/h.
        pres (Optional[float]): Atmospheric pressure in hPa.
        tsun (Optional[float]): Total sunshine duration in minutes.
        coco (Optional[WeatherCode]): Weather condition code.
    """
    
    time: datetime = Field(..., description="Timestamp of the weather observation")
    temp: Optional[float] = Field(None, description="Temperature in Celsius")
    dwpt: Optional[float] = Field(None, description="Dew point in Celsius")
    rhum: Optional[float] = Field(None, description="Relative humidity in percentage")
    prcp: Optional[float] = Field(None, description="Precipitation in millimeters")
    snow: Optional[float] = Field(None, description="Snowfall in millimeters")
    wdir: Optional[float] = Field(None, description="Wind direction in degrees (0 to 360)")
    wspd: Optional[float] = Field(None, description="Wind speed in km/h")
    wpgt: Optional[float] = Field(None, description="Maximum wind gust in km/h")
    pres: Optional[float] = Field(None, description="Atmospheric pressure in hPa")
    tsun: Optional[float] = Field(None, description="Total sunshine duration in minutes")
    coco: Optional[WeatherCode] = Field(None, description="Weather condition code")

class WeatherSearchModelData(BaseModel):
    """
    WeatherSearchModelData represents a reduced set of weather data for searching or querying purposes.

    Attributes:
        time (datetime): Timestamp of the weather observation.
        dwpt (Optional[float]): Dew point in Celsius.
        rhum (Optional[float]): Relative humidity in percentage.
        prcp (Optional[float]): Precipitation in millimeters.
        snow (Optional[float]): Snowfall in millimeters.
        wdir (Optional[float]): Wind direction in degrees (0 to 360).
        wspd (Optional[float]): Wind speed in km/h.
        wpgt (Optional[float]): Maximum wind gust in km/h.
        pres (Optional[float]): Atmospheric pressure in hPa.
        tsun (Optional[float]): Total sunshine duration in minutes.
        coco (Optional[int]): Weather condition code as an integer.
    """
    
    time: datetime = Field(..., description="Timestamp of the weather observation")
    dwpt: Optional[float] = Field(None, description="Dew point in Celsius")
    rhum: Optional[float] = Field(None, description="Relative humidity in percentage")
    prcp: Optional[float] = Field(None, description="Precipitation in millimeters")
    snow: Optional[float] = Field(None, description="Snowfall in millimeters")
    wdir: Optional[float] = Field(None, description="Wind direction in degrees (0 to 360)")
    wspd: Optional[float] = Field(None, description="Wind speed in km/h")
    wpgt: Optional[float] = Field(None, description="Maximum wind gust in km/h")
    pres: Optional[float] = Field(None, description="Atmospheric pressure in hPa")
    tsun: Optional[float] = Field(None, description="Total sunshine duration in minutes")
    coco: Optional[int] = Field(None, description="Weather condition code as an integer")
