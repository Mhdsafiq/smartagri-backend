from flask import Blueprint, request, jsonify
import requests
import logging

logger = logging.getLogger(__name__)
weather_bp = Blueprint('weather', __name__)

@weather_bp.route('/weather', methods=['POST'])
def get_weather():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')

        if not lat or not lon:
            return jsonify({'error': 'Missing latitude or longitude'}), 400

        # NASA POWER API
        # Fetch T2M (Temperature at 2 Meters), PRECTOTCORR (Precipitation Corrected), RH2M (Relative Humidity at 2 Meters)
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'T2M,PRECTOTCORR,RH2M',
            'community': 'AG',
            'longitude': lon,
            'latitude': lat,
            'start': '20230101', # Mocking recent historical data as forecast is not available on free tier easily without intricate setup
            'end': '20230105',
            'format': 'JSON'
        }

        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
             logger.error(f"NASA API Error: {response.text}")
             return jsonify({'error': 'Failed to fetch weather data from NASA API'}), 502

        nasa_data = response.json()
        
        # Process and average the data
        properties = nasa_data.get('properties', {}).get('parameter', {})
        t2m = properties.get('T2M', {})
        prectot = properties.get('PRECTOTCORR', {})
        rh2m = properties.get('RH2M', {})

        # Calculate averages (ignoring -999 which is NASA's null value)
        def get_avg(data_dict):
            values = [v for v in data_dict.values() if v != -999]
            return sum(values) / len(values) if values else 0

        avg_temp = get_avg(t2m)
        avg_rainfall = get_avg(prectot)
        avg_humidity = get_avg(rh2m)

        return jsonify({
            'temperature': round(avg_temp, 2),
            'rainfall': round(avg_rainfall, 2),
            'humidity': round(avg_humidity, 2),
            'source': 'NASA POWER API'
        })

    except Exception as e:
        logger.error(f"Error in weather route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@weather_bp.route('/geocode', methods=['POST'])
def geocode_location():
    try:
        data = request.get_json()
        city = data.get('city')

        if not city:
            return jsonify({'error': 'Missing city name'}), 400

        # OpenStreetMap Nominatim API
        url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
        headers = {
            'User-Agent': 'SmartAgroAI/1.0'
        }

        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
             return jsonify({'error': 'Failed to fetch location data'}), 502

        results = response.json()
        
        if not results:
             return jsonify({'error': 'Location not found'}), 404

        location = results[0]
        
        return jsonify({
            'latitude': location.get('lat'),
            'longitude': location.get('lon'),
            'display_name': location.get('display_name')
        })

    except Exception as e:
        logger.error(f"Error in geocode route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
