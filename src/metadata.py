import PIL.Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime

def get_exif_data(image_path):
    """
    Extracts EXIF data from an image.
    Returns a dictionary with 'date', 'model', 'lat', 'lon'.
    """
    try:
        image = PIL.Image.open(image_path)
        exif = image._getexif()
        
        if not exif:
            return {}

        data = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                data['date'] = value
            elif tag == 'Model':
                data['model'] = str(value)
            elif tag == 'Make':
                data['make'] = str(value)
            elif tag == 'FNumber':
                data['f_number'] = float(value) if isinstance(value, (int, float)) else str(value)
            elif tag == 'ISOSpeedRatings':
                data['iso'] = str(value)
            elif tag == 'GPSInfo':
                lat, lon = get_lat_lon(value)
                if lat and lon:
                    data['lat'] = lat
                    data['lon'] = lon
        
        return data
    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return {}

def get_lat_lon(gps_info):
    """
    Decodes the GPSInfo tag.
    Returns (lat, lon) as floats or (None, None).
    """
    try:
        if not gps_info:
            return None, None

        def _convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)

        lat_raw = gps_info.get(2) # GPSLatitude
        lat_ref = gps_info.get(1) # GPSLatitudeRef
        lon_raw = gps_info.get(4) # GPSLongitude
        lon_ref = gps_info.get(3) # GPSLongitudeRef

        if lat_raw and lat_ref and lon_raw and lon_ref:
            lat = _convert_to_degrees(lat_raw)
            if lat_ref != 'N':
                lat = -lat

            lon = _convert_to_degrees(lon_raw)
            if lon_ref != 'E':
                lon = -lon
            
            return lat, lon
            
        return None, None
    except Exception:
        return None, None
