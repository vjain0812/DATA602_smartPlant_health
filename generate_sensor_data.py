"""
Generate mock sensor data for LSTM training
Simulates: temperature, humidity, soil moisture, light, health score
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sensor_data(days=60, freq='1H', plants=10):
    """
    Generate realistic mock sensor data for multiple plants
    
    Args:
        days: Number of days of data
        freq: Frequency of readings (e.g., '1H' for hourly)
        plants: Number of different plants to simulate
    
    Returns:
        DataFrame with sensor readings
    """
    
    all_data = []
    
    for plant_id in range(plants):
        start_date = datetime.now() - timedelta(days=days)
        date_range = pd.date_range(start=start_date, periods=days*24, freq=freq)
        
        # Get hour of day for patterns
        hours = np.array([d.hour for d in date_range])
        days_num = np.arange(len(hours))
        
        # Temperature (°C) - varies by time of day
        base_temp = 22 + np.random.uniform(-2, 2)  # Each plant slightly different
        daily_variation = 8 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
        seasonal_trend = 2 * np.sin(2 * np.pi * days_num / (days*24))
        temperature = base_temp + daily_variation + seasonal_trend + np.random.normal(0, 1, len(hours))
        temperature = np.clip(temperature, 10, 35)
        
        # Humidity (%) - inverse relationship with temperature
        base_humidity = 65 + np.random.uniform(-5, 5)
        humidity = base_humidity - daily_variation/2 + np.random.normal(0, 3, len(hours))
        humidity = np.clip(humidity, 30, 90)
        
        # Soil Moisture (%) - decreases over time, spikes when watered
        soil_moisture = np.zeros(len(hours))
        soil_moisture[0] = np.random.uniform(70, 85)
        
        watering_threshold = 35 + np.random.uniform(-5, 5)  # Each plant different
        
        for i in range(1, len(hours)):
            # Natural decrease (faster when hot)
            evaporation_rate = 0.3 + 0.2 * (temperature[i] - 20) / 15
            soil_moisture[i] = soil_moisture[i-1] - evaporation_rate - np.random.uniform(0, 0.2)
            
            # Water when too dry (simulate watering schedule)
            if soil_moisture[i] < watering_threshold:
                soil_moisture[i] = np.random.uniform(75, 85)
        
        soil_moisture = np.clip(soil_moisture, 15, 90)
        
        # Light Intensity (lux) - 0 at night, high during day
        light_peak = 900 + np.random.uniform(-100, 100)
        light = np.maximum(0, light_peak * np.sin(2 * np.pi * (hours - 6) / 24))
        
        # Add clouds/weather variation
        weather_factor = np.random.uniform(0.7, 1.0, len(hours))
        light = light * weather_factor + np.random.normal(0, 30, len(hours))
        light = np.clip(light, 0, 1000)
        
        # Health Score (0-100) - affected by environmental conditions
        health_base = 85
        
        # Penalties for poor conditions
        moisture_penalty = np.where(soil_moisture < 30, 20, 0)
        moisture_penalty += np.where(soil_moisture > 80, 5, 0)
        
        temp_penalty = np.where(temperature > 32, 15, 0)
        temp_penalty += np.where(temperature < 12, 15, 0)
        
        light_penalty = np.where(light < 200, 10, 0) * (hours > 8) * (hours < 20)
        
        health_score = health_base - moisture_penalty - temp_penalty - light_penalty
        health_score = health_score + np.random.normal(0, 3, len(hours))
        health_score = np.clip(health_score, 0, 100)
        
        # Add some disease events (random drops in health)
        if np.random.random() > 0.7:  # 30% of plants get disease
            disease_start = np.random.randint(len(hours) // 2, len(hours))
            disease_duration = np.random.randint(48, 168)  # 2-7 days
            disease_end = min(disease_start + disease_duration, len(hours))
            health_score[disease_start:disease_end] -= np.random.uniform(20, 40)
            health_score = np.clip(health_score, 0, 100)
        
        # Watering Needed (binary) - predict 24 hours ahead
        watering_needed = np.zeros(len(hours), dtype=int)
        for i in range(len(hours) - 24):
            # Will need water in next 24 hours?
            future_moisture = soil_moisture[i:i+24]
            if np.any(future_moisture < watering_threshold + 5):
                watering_needed[i] = 1
        
        # NPK levels (Nitrogen, Phosphorus, Potassium) - decrease slowly
        npk_n = 80 - days_num * 0.3 + np.random.normal(0, 2, len(hours))
        npk_p = 75 - days_num * 0.25 + np.random.normal(0, 2, len(hours))
        npk_k = 85 - days_num * 0.35 + np.random.normal(0, 2, len(hours))
        
        npk_n = np.clip(npk_n, 20, 100)
        npk_p = np.clip(npk_p, 20, 100)
        npk_k = np.clip(npk_k, 20, 100)
        
        # Create DataFrame for this plant
        plant_data = pd.DataFrame({
            'plant_id': plant_id,
            'timestamp': date_range,
            'temperature': temperature,
            'humidity': humidity,
            'soil_moisture': soil_moisture,
            'light_intensity': light,
            'npk_nitrogen': npk_n,
            'npk_phosphorus': npk_p,
            'npk_potassium': npk_k,
            'health_score': health_score,
            'watering_needed': watering_needed
        })
        
        all_data.append(plant_data)
    
    # Combine all plants
    df = pd.concat(all_data, ignore_index=True)
    
    return df

def plot_sensor_data(df, plant_id=0, save_path='sensor_data_visualization.png'):
    """Visualize sensor data for one plant"""
    
    plant_data = df[df['plant_id'] == plant_id].copy()
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Plant {plant_id} - Sensor Data Over Time', fontsize=16, fontweight='bold')
    
    # Temperature
    axes[0, 0].plot(plant_data['timestamp'], plant_data['temperature'], linewidth=1, color='red')
    axes[0, 0].set_title('Temperature (°C)')
    axes[0, 0].set_ylabel('Temperature')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Humidity
    axes[0, 1].plot(plant_data['timestamp'], plant_data['humidity'], linewidth=1, color='blue')
    axes[0, 1].set_title('Humidity (%)')
    axes[0, 1].set_ylabel('Humidity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Soil Moisture
    axes[1, 0].plot(plant_data['timestamp'], plant_data['soil_moisture'], linewidth=1, color='brown')
    axes[1, 0].axhline(y=35, color='red', linestyle='--', label='Watering Threshold')
    axes[1, 0].set_title('Soil Moisture (%)')
    axes[1, 0].set_ylabel('Soil Moisture')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Light Intensity
    axes[1, 1].plot(plant_data['timestamp'], plant_data['light_intensity'], linewidth=1, color='orange')
    axes[1, 1].set_title('Light Intensity (lux)')
    axes[1, 1].set_ylabel('Light')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Health Score
    axes[2, 0].plot(plant_data['timestamp'], plant_data['health_score'], linewidth=1.5, color='green')
    axes[2, 0].set_title('Plant Health Score')
    axes[2, 0].set_ylabel('Health Score')
    axes[2, 0].set_ylim(0, 100)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Watering Events
    watering_events = plant_data[plant_data['watering_needed'] == 1]
    axes[2, 1].scatter(watering_events['timestamp'], 
                       watering_events['soil_moisture'],
                       c='red', alpha=0.5, s=20)
    axes[2, 1].plot(plant_data['timestamp'], plant_data['soil_moisture'], 
                   linewidth=1, color='brown', alpha=0.3)
    axes[2, 1].set_title('Watering Predictions')
    axes[2, 1].set_ylabel('Soil Moisture')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mock sensor data')
    parser.add_argument('--days', type=int, default=60, help='Days of data to generate')
    parser.add_argument('--plants', type=int, default=10, help='Number of plants')
    parser.add_argument('--output', type=str, default='data/timeseries/sensor_data.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" "*20 + "SENSOR DATA GENERATOR")
    print("="*70)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print(f"\n📊 Generating sensor data...")
    print(f"   Days: {args.days}")
    print(f"   Plants: {args.plants}")
    print(f"   Frequency: Hourly")
    
    df = generate_sensor_data(days=args.days, plants=args.plants)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"\n✅ Data generated successfully!")
    print(f"   Total records: {len(df):,}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"   Saved to: {args.output}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("DATA STATISTICS")
    print(f"{'='*70}")
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nSensor readings per plant: {len(df[df['plant_id'] == 0])}")
    print(f"Total watering events: {df['watering_needed'].sum()}")
    print(f"Average health score: {df['health_score'].mean():.1f}")
    
    print(f"\nEnvironmental ranges:")
    print(f"  Temperature: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
    print(f"  Humidity: {df['humidity'].min():.1f}% to {df['humidity'].max():.1f}%")
    print(f"  Soil Moisture: {df['soil_moisture'].min():.1f}% to {df['soil_moisture'].max():.1f}%")
    print(f"  Light: {df['light_intensity'].min():.0f} to {df['light_intensity'].max():.0f} lux")
    
    print(f"\n📈 Sample data:")
    print(df.head(10).to_string())
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    plot_sensor_data(df, plant_id=0)
    
    print(f"\n🎉 Ready for LSTM training!")
    print(f"\nNext: python src/train_lstm.py --data {args.output}")