from flask import Flask, render_template, request, jsonify
import os
import pynmea2
import pandas as pd
import numpy as np
from scipy.stats import zscore
from math import radians, sin, cos, sqrt, atan2
import csv
from datetime import datetime

app = Flask(__name__)

# Opprett uploads-mappe hvis den ikke eksisterer
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def parse_nmea_file(file_path):
    """
    Parser NMEA-fil og ekstraherer GPS-koordinater.
    """
    coordinates = []
    valid_sentences = ['$GPGGA', '$GPRMC', '$GNRMC', '$GNGGA']
    
    with open(file_path, 'r') as file:
        for line in file:
            try:
                line = line.strip()
                if any(line.startswith(sentence) for sentence in valid_sentences):
                    msg = pynmea2.parse(line)
                    
                    if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                        if msg.latitude and msg.longitude:
                            # Legg til datetime.now() som timestamp hvis msg ikke har timestamp
                            timestamp = msg.timestamp if hasattr(msg, 'timestamp') else pd.Timestamp.now().time()
                            coordinates.append({
                                'Latitude': msg.latitude,
                                'Longitude': msg.longitude,
                                'Timestamp': pd.Timestamp.combine(pd.Timestamp.now().date(), timestamp)  # Kombiner dagens dato med tidspunkt
                            })
            except pynmea2.ParseError as e:
                print(f"Kunne ikke parse linje: {line}. Feil: {str(e)}")
                continue
            except Exception as e:
                print(f"Uventet feil ved parsing av linje: {line}. Feil: {str(e)}")
                continue
    
    if not coordinates:
        print("Ingen koordinater funnet i filen")
        return pd.DataFrame()
        
    df = pd.DataFrame(coordinates)
    print(f"Fant {len(df)} GPS-punkter")
    return df

def parse_csv_file(file_path):
    """
    Parser CSV-fil med GPS-data.
    Støtter flere mulige kolonnenavn for lat/lon/tid.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Mulige kolonnenavn for hver type
        lat_columns = ['latitude', 'lat', 'breddegrad', 'Latitude', 'Lat']
        lon_columns = ['longitude', 'lon', 'lengdegrad', 'Longitude', 'Lon', 'Long']
        time_columns = ['timestamp', 'time', 'tid', 'Timestamp', 'Time']
        
        # Finn matchende kolonner
        lat_col = next((col for col in lat_columns if col in df.columns), None)
        lon_col = next((col for col in lon_columns if col in df.columns), None)
        time_col = next((col for col in time_columns if col in df.columns), None)
        
        if not (lat_col and lon_col):
            print(f"Fant ikke latitude/longitude kolonner. Tilgjengelige kolonner: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Lag ny dataframe med standardiserte kolonnenavn
        result_df = pd.DataFrame()
        result_df['Latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
        result_df['Longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
        
        # Håndter timestamp
        if time_col:
            try:
                result_df['Timestamp'] = pd.to_datetime(df[time_col])
            except:
                print(f"Kunne ikke parse timestamp kolonne: {time_col}")
                result_df['Timestamp'] = pd.date_range(
                    start=pd.Timestamp.now(), 
                    periods=len(df), 
                    freq='S'
                )
        else:
            # Hvis ingen timestamp kolonne finnes, lag sekvensielle timestamps
            result_df['Timestamp'] = pd.date_range(
                start=pd.Timestamp.now(), 
                periods=len(df), 
                freq='S'
            )
        
        # Fjern ugyldige verdier
        result_df = result_df.dropna()
        
        # Valider verdier
        result_df = result_df[
            (result_df['Latitude'].between(-90, 90)) & 
            (result_df['Longitude'].between(-180, 180))
        ]
        
        if len(result_df) == 0:
            print("Ingen gyldige GPS-punkter funnet i CSV-filen")
            return pd.DataFrame()
            
        print(f"Fant {len(result_df)} gyldige GPS-punkter")
        return result_df
        
    except Exception as e:
        print(f"Feil under parsing av CSV: {str(e)}")
        return pd.DataFrame()

class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_estimate = 1.0

    def update(self, measurement):
        prediction = self.estimate
        prediction_error = self.error_estimate + self.process_variance
        
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * prediction_error
        
        return self.estimate

def calculate_speed(df):
    """Beregner hastighet mellom GPS-punkter i km/t"""
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Jordens radius i km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    speeds = []
    for i in range(len(df)-1):
        dist = haversine_distance(
            df['Latitude'].iloc[i], df['Longitude'].iloc[i],
            df['Latitude'].iloc[i+1], df['Longitude'].iloc[i+1]
        )
        time_diff = (df['Timestamp'].iloc[i+1] - df['Timestamp'].iloc[i]).total_seconds() / 3600
        speed = dist/time_diff if time_diff > 0 else 0
        speeds.append(speed)
    speeds.append(0)  # Legg til 0 for siste punkt
    
    return speeds

def process_gps_data(df):
    """Prosesserer GPS-data med Kalman-filter og anomalideteksjon"""
    # Konverter timestamp til datetime hvis det ikke allerede er det
    if isinstance(df['Timestamp'].iloc[0], str):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Beregn hastighet
    df['Speed'] = calculate_speed(df)
    
    # Bruk Kalman-filter på posisjon
    kalman_lat = SimpleKalmanFilter()
    kalman_lon = SimpleKalmanFilter()
    df['Filtered_Latitude'] = df['Latitude'].apply(kalman_lat.update)
    df['Filtered_Longitude'] = df['Longitude'].apply(kalman_lon.update)
    
    # Beregn z-scores for hastighet og posisjon
    df['Speed_Zscore'] = zscore(df['Speed'])
    df['Position_Zscore'] = zscore(np.sqrt(
        (df['Filtered_Latitude'] - df['Latitude'])**2 + 
        (df['Filtered_Longitude'] - df['Longitude'])**2
    ))
    
    # Merk anomalier
    df['Is_Anomaly'] = (
        (abs(df['Speed_Zscore']) > 3) |  # Unormal hastighet
        (abs(df['Position_Zscore']) > 3)  # Unormal posisjon
    )
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Ingen fil lastet opp'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Ingen fil valgt'}), 400
        
    if not file.filename.endswith(('.txt', '.nmea', '.csv')):
        return jsonify({'error': 'Kun .txt, .nmea eller .csv filer er tillatt'}), 400
    
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Velg parser basert på filtype
        if file.filename.endswith('.csv'):
            gps_data = parse_csv_file(file_path)
        else:
            gps_data = parse_nmea_file(file_path)
            
        os.remove(file_path)
        
        if len(gps_data) == 0:
            return jsonify({'error': 'Ingen gyldige GPS-data funnet i filen'}), 400
            
        processed_data = process_gps_data(gps_data)
        
        return jsonify({
            'message': f'Fil prosessert. Fant {len(processed_data)} GPS-punkter.',
            'stats': {
                'total_points': len(processed_data),
                'anomalies': int(processed_data['Is_Anomaly'].sum()),
                'avg_speed': float(processed_data['Speed'].mean()),
                'max_speed': float(processed_data['Speed'].max())
            },
            'data': {
                'points': processed_data.apply(
                    lambda row: {
                        'lat': float(row['Latitude']),
                        'lon': float(row['Longitude']),
                        'time': str(row['Timestamp']),
                        'speed': float(row['Speed']),
                        'is_anomaly': bool(row['Is_Anomaly'])
                    }, 
                    axis=1
                ).tolist()
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Feil under prosessering: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 