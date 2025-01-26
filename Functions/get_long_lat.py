from geopy.geocoders import Nominatim
import time

def get_lat_lon(cities):
     # Initialize Nominatim geocoder
    geolocator = Nominatim(user_agent="your_name_script_name_v1")

    city_coordinates = {}
    # Save the results to a text file
    with open("city_coordinates.txt", "w", encoding='utf-8') as file:
        for city in cities:
            try:
                # Geocode the city to get latitude and longitude
                location = geolocator.geocode(city, timeout=10)
                
                if location:
                    city_coordinates[city] = (location.latitude, location.longitude)
                    file.write(f"{city}: ({location.latitude:.2f}, {location.longitude:.2f})\n")
                    print(f"{city}: ({location.latitude:.2f}, {location.longitude:.2f})")
                else:
                    city_coordinates[city] = None
                    print(f"{city}: Not found")
                
                # To avoid hitting the API too frequently, we can pause for a second
                time.sleep(4)
            except Exception as e:
                file.write(f"{city}: \n")
                print(f"Error occurred for {city}: {e}")
                city_coordinates[city] = None
                continue
    return city_coordinates
