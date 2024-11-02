#app\usecases\usecase_create_data_puppeto4.py
from datetime import datetime
from meteostat import Point, Hourly  # type: ignore

def usecase_create_data_puppeto4():
    """
    Fetches and saves hourly weather data for Grenoble from 2018 to 2024.
    Data is collected from Meteostat and saved as a JSON file with key weather indicators.
    """

    # Define the desired time period
    start = datetime(2018, 1, 1)
    end = datetime(2024, 10, 1)

    # Define the location for Grenoble using coordinates
    grenoble = Point(45.1885, 5.7245)

    # Retrieve hourly weather data for the specified location and time range
    data = Hourly(grenoble, start, end)
    data = data.fetch()

    # Reset index to convert 'time' from index to a column in the DataFrame
    data = data.reset_index()

    # Convert the DataFrame to JSON format with ISO date formatting
    json_data = data.to_json(orient='records', date_format='iso')

    # Save the JSON data to a file
    with open('grenoble_weather_data.json', 'w') as f:
        f.write(json_data)

    # Metadata for the JSON fields
    # time : Timestamp of the observation.
    # temp : Temperature in degrees Celsius.
    # dwpt : Dew point in degrees Celsius.
    # rhum : Relative humidity in percentage.
    # prcp : Precipitation in millimeters.
    # snow : Snowfall in millimeters.
    # wdir : Wind direction in degrees.
    # wspd : Wind speed in km/h.
    # wpgt : Max wind gust in km/h.
    # pres : Atmospheric pressure in hPa.
    # tsun : Sunshine duration in minutes.
    # coco : Weather condition code.

    print("Data has been saved to 'grenoble_weather_data.json'")
    
    return True
