from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import IntEnum

class WeatherCode(IntEnum):
    CLEAR = 0                  # Ciel clair
    MAINLY_CLEAR = 1           # Principalement clair
    PARTLY_CLOUDY = 2          # Partiellement nuageux
    OVERCAST = 3               # Couvert
    FOG = 4                    # Brouillard
    DRIZZLE = 5                # Bruine
    FREEZING_DRIZZLE = 6       # Bruine verglaçante
    RAIN = 7                   # Pluie
    FREEZING_RAIN = 8          # Pluie verglaçante
    SNOW_FALL = 9              # Chute de neige
    SNOW_GRAINS = 10           # Grains de neige
    RAIN_SHOWER = 11           # Averse de pluie
    SNOW_SHOWER = 12           # Averse de neige
    THUNDERSTORM = 13          # Orage
    THUNDERSTORM_WITH_HAIL = 14 # Orage avec grêle
    OTHER = 15 # Autre
    OTHER1 = 16 # Autre
    OTHER2 = 17 # Autre
    OTHER3 = 18 # Autre
    OTHER4 = 19 # Autre
    OTHER5 = 20 # Autre
    OTHER6 = 21 # Autre
    OTHER7 = 22 # Autre
    OTHER8 = 23 # Autre

class WeatherModelData(BaseModel):
    time: datetime = Field(..., description="Horodatage de l'observation")
    temp: Optional[float] = Field(None, description="Température en degrés Celsius")
    dwpt: Optional[float] = Field(None, description="Point de rosée en degrés Celsius")
    rhum: Optional[float] = Field(None, description="Humidité relative en pourcentage")
    prcp: Optional[float] = Field(None, description="Précipitations en millimètres")
    snow: Optional[float] = Field(None, description="Chute de neige en millimètres")
    wdir: Optional[float] = Field(None, description="Direction du vent en degrés (0 à 360)")
    wspd: Optional[float] = Field(None, description="Vitesse du vent en km/h")
    wpgt: Optional[float] = Field(None, description="Rafale de vent maximale en km/h")
    pres: Optional[float] = Field(None, description="Pression atmosphérique en hPa")
    tsun: Optional[float] = Field(None, description="Durée totale d'ensoleillement en minutes")
    coco: Optional[WeatherCode] = Field(None, description="Code de condition météorologique")

class WeatherSearchModelData(BaseModel):
    time: datetime = Field(..., description="Horodatage de l'observation")
    dwpt: Optional[float] = Field(None, description="Point de rosée en degrés Celsius")
    rhum: Optional[float] = Field(None, description="Humidité relative en pourcentage")
    prcp: Optional[float] = Field(None, description="Précipitations en millimètres")
    snow: Optional[float] = Field(None, description="Chute de neige en millimètres")
    wdir: Optional[float] = Field(None, description="Direction du vent en degrés (0 à 360)")
    wspd: Optional[float] = Field(None, description="Vitesse du vent en km/h")
    wpgt: Optional[float] = Field(None, description="Rafale de vent maximale en km/h")
    pres: Optional[float] = Field(None, description="Pression atmosphérique en hPa")
    tsun: Optional[float] = Field(None, description="Durée totale d'ensoleillement en minutes")
    coco: Optional[int] = Field(None, description="Code de condition météorologique")