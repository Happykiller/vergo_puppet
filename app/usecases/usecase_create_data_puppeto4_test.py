# app/usecases/usecase_create_data_puppeto4_test.py
import json
import pytest
from unittest.mock import patch, MagicMock
from app.usecases.usecase_create_data_puppeto4 import usecase_create_data_puppeto4

# Test case for verifying the correct creation of data and file structure
@patch('app.usecases.usecase_create_data_puppeto4.Hourly')
@patch('app.usecases.usecase_create_data_puppeto4.open')
def test_usecase_create_data_puppeto4(mock_open, mock_hourly):
    # Mock data to simulate Meteostat's response
    mock_data = MagicMock()
    mock_data.reset_index.return_value = mock_data  # To handle the reset_index() chain
    mock_data.to_json.return_value = json.dumps([{
        "time": "2018-01-01T00:00:00Z",
        "temp": 2.5,
        "dwpt": -3.0,
        "rhum": 85,
        "prcp": 0.0,
        "snow": 0.0,
        "wdir": 180,
        "wspd": 10,
        "wpgt": 15,
        "pres": 1013,
        "tsun": 0,
        "coco": 3
    }])

    # Configure mock_hourly to return mock_data on fetch
    mock_instance = MagicMock()
    mock_instance.fetch.return_value = mock_data
    mock_hourly.return_value = mock_instance

    # Mock the file write
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    # Run the function
    result = usecase_create_data_puppeto4()

    # Assertions to verify behavior
    assert result is True, "The function should return True on success"
    
    # Verify that Hourly was called with expected coordinates and date range
    mock_hourly.assert_called_once()
    mock_instance.fetch.assert_called_once()  # Ensure data fetch was called

    # Check if the JSON data was written to a file
    mock_open.assert_called_once_with('grenoble_weather_data.json', 'w')
    mock_file.write.assert_called_once()  # Verify that write was called
    written_data = mock_file.write.call_args[0][0]
    
    # Verify JSON structure by loading the written data
    parsed_data = json.loads(written_data)
    assert isinstance(parsed_data, list), "The JSON output should be a list of records"
    assert "time" in parsed_data[0], "The output should include 'time' field"
    assert "temp" in parsed_data[0], "The output should include 'temp' field"
    assert "dwpt" in parsed_data[0], "The output should include 'dwpt' field"
    assert "rhum" in parsed_data[0], "The output should include 'rhum' field"
    assert "prcp" in parsed_data[0], "The output should include 'prcp' field"
    assert "wdir" in parsed_data[0], "The output should include 'wdir' field"
