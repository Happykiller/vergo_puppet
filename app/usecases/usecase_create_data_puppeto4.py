from datetime import datetime
import pandas as pd
from meteostat import Point, Hourly  # type: ignore

def usecase_create_data_puppeto4():
  # Définir la période souhaitée
  start = datetime(2018, 1, 1)
  end = datetime(2024, 10, 1)

  # Définir la localisation de Grenoble
  grenoble = Point(45.1885, 5.7245)

  # Récupérer les données horaires
  data = Hourly(grenoble, start, end)
  data = data.fetch()

  # Réinitialiser l'index pour obtenir 'time' comme colonne
  data = data.reset_index()

  # Convertir le DataFrame en JSON
  json_data = data.to_json(orient='records', date_format='iso')

  # Sauvegarder les données JSON dans un fichier
  with open('grenoble_weather_data.json', 'w') as f:
      f.write(json_data)

  #time : Horodatage de l'observation.
  #temp : Température en degrés Celsius.
  #dwpt : Point de rosée en degrés Celsius.
  #rhum : Humidité relative en pourcentage.
  #prcp : Précipitations en millimètres.
  #snow : Chute de neige en millimètres.
  #wdir : Direction du vent en degrés.
  #wspd : Vitesse du vent en km/h.
  #wpgt : Rafale de vent maximale en km/h.
  #pres : Pression atmosphérique en hPa.
  #tsun : Durée d'ensoleillement en minutes.
  #coco : Code de condition météorologique.

  print("Les données ont été sauvegardées dans 'grenoble_weather_data.json'")
    
  return True