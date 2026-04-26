import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib



class FuelConsumptionPredictor(nn.Module):
    """Deep Learning model for fuel consumption prediction"""
    def __init__(self):
        super(FuelConsumptionPredictor, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class TrajectoryOptimizer(nn.Module):
    """DL model for optimal trajectory calculation"""
    def __init__(self):
        super(TrajectoryOptimizer, self).__init__()
        self.lstm = nn.LSTM(6, 128, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 3)  # x, y, z velocities
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class MissionRiskDetector:
    """ML model for anomaly detection"""
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_risk(self, features):
        if not self.is_trained:
            return 0.05  # Default low risk
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]

# Global models
fuel_model = FuelConsumptionPredictor()
trajectory_model = TrajectoryOptimizer()
risk_detector = MissionRiskDetector()

def train_models():
    """Train all models with simulated space mission data"""
    print("Training AI models...")
    
    # Fuel model training
    optimizer = optim.Adam(fuel_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Realistic training data: [speed, distance, fuel_used, time, autopilot]
    X_train = torch.randn(1000, 5) * torch.tensor([10000, 30, 50000, 365, 1.0])
    y_train = (X_train[:, 0] * 0.001 + X_train[:, 1] * 0.0001 + torch.randn(1000, 1) * 5000).clamp(0, 100000)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = fuel_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Risk detector training
    X_risk = np.random.randn(500, 8) * np.array([10000, 100000, 30, 0.1, 0.1, 0.1, 1, 1])
    y_risk = np.abs(np.random.randn(500)) * 0.3 + 0.05
    risk_detector.train(X_risk, y_risk)
    
    print("✅ All models trained successfully!")

# Train models on startup
train_models()

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class SpaceMissionSimulator:
    def __init__(self):
        self.planets = {
            'Earth': {'distance': 0, 'radius': 6371, 'color': '#4A90E2', 'gravity': 9.81},
            'Mars': {'distance': 2.25e8, 'radius': 3390, 'color': '#CD5C5C', 'gravity': 3.71},
            'Jupiter': {'distance': 7.78e8, 'radius': 69911, 'color': '#D8CA9D', 'gravity': 24.79},
            'Saturn': {'distance': 1.427e9, 'radius': 58232, 'color': '#FAD5A5', 'gravity': 10.44},
            'Uranus': {'distance': 2.871e9, 'radius': 25362, 'color': '#4FD0E7', 'gravity': 8.69}
        }
        self.reset_mission()
    
    def reset_mission(self):
        self.mission_active = False
        self.current_time = 0
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.fuel = 100000
        self.max_fuel = 100000
        self.speed = 0
        self.distance_traveled = 0
        self.logs = []
        self.crew = {
            'commander': 'Dr. Elena Voss',
            'pilot': 'Capt. Marcus Reed',
            'engineer': 'Dr. Aisha Khan',
            'scientist': 'Prof. Liam Chen'
        }
        self.status = "Grounded - Ready for Launch"
    
    def launch(self):
        self.mission_active = True
        self.status = "LAUNCH INITIATED - Escape Velocity Achieved"
        self.velocity = np.array([0.1, 0.05, 11.2])  # km/s with slight orbital inclination
        self.logs.append({"timestamp": self.current_time, "event": "LAUNCH", "status": "SUCCESS"})
    
    def update_physics(self, dt=3600):  # 1 hour steps
        if not self.mission_active:
            return
        
        target_distance = self.planets['Uranus']['distance']
        
        # Mission phases
        if self.current_time < 300:  # Launch burn (5 minutes)
            thrust = 5000 * (1 - self.current_time/300)  # Decreasing thrust
            fuel_consumed = thrust * dt / 10000
            if self.fuel > fuel_consumed:
                self.fuel -= fuel_consumed
                self.velocity += np.array([0.001, 0.0005, 0.01]) * dt
        
        elif self.distance_traveled < target_distance * 0.3:  # Mid-course corrections
            if random.random() < 0.01:  # Occasional corrections
                self.velocity += np.array([0.0001, 0.00005, 0.0002])
        
        else:  # Coast + deceleration
            # Orbital decay simulation
            r = np.linalg.norm(self.position)
            if r > 1e6:
                accel = -3.986e14 / r**2 * np.array([self.position[0]/r, self.position[1]/r, self.position[2]/r])
                self.velocity += accel * dt / 3600
        
        # Update position
        self.position += self.velocity * dt
        self.distance_traveled = np.linalg.norm(self.position)
        self.speed = np.linalg.norm(self.velocity) * 3600  # km/h
        self.current_time += dt
        
        # Update status dynamically
        progress = self.distance_traveled / target_distance
        if progress > 0.95:
            self.status = "URANUS ORBIT - Landing Sequence Initiated"
        elif progress > 0.7:
            self.status = "Saturn Flyby Complete - Final Approach"
        elif progress > 0.4:
            self.status = "Jupiter Gravity Assist - Course Correction"
        elif progress > 0.1:
            self.status = "Trans-Uranus Injection - Nominal"
    
    def get_telemetry(self):
        uranus_distance = self.planets['Uranus']['distance']
        distance_remaining = max(0, uranus_distance - self.distance_traveled)
        avg_speed_kms = self.speed / 3600
        eta_days = distance_remaining / (avg_speed_kms * 86400) if avg_speed_kms > 0 else float('inf')
        
        return {
            'speed': self.speed,
            'fuel_remaining': self.fuel,
            'fuel_consumed': self.max_fuel - self.fuel,
            'fuel_return_needed': max(0, 65000 - self.fuel),
            'distance_covered': self.distance_traveled / 1e9,
            'distance_remaining': distance_remaining / 1e9,
            'eta_months': eta_days / 30,
            'return_duration_years': 8.5,
            'altitude': np.linalg.norm(self.position),
            'mission_time_days': self.current_time / 86400
        }

import requests
import json

class NASADataAPI:
    def __init__(self):
        self.apod_key = "DEMO_KEY"  # Free demo key works!
    
    def get_apod(self):
        """Astronomy Picture of the Day"""
        try:
            url = f"https://api.nasa.gov/planetary/apod?api_key={self.apod_key}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        # Fallback
        return {
            "title": "Deep Space Nebula",
            "url": "https://images-assets.nasa.gov/image/PIA06423/PIA06423~orig.jpg",
            "explanation": "Stunning view of deep space"
        }
    
    def get_mars_weather(self):
        """InSight Mars Weather"""
        try:
            url = f"https://api.nasa.gov/insight_weather/?api_key={self.apod_key}&feedtype=json&ver=1.0"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"sol": 1000, "AT": {"mx": -20, "mn": -80}, "status": "Nominal"}
    
    def get_neo_feed(self):
        """Near Earth Objects"""
        try:
            url = f"https://api.nasa.gov/neo/rest/v1/feed?api_key={self.apod_key}&start_date=2024-01-01&end_date=2024-01-07"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {"element_count": 25, "near_earth_objects": {}}

# Global instance
nasa_api = NASADataAPI()

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def create_mission_dashboard_charts(telemetry_data, simulation_logs):
    """NASA-style mission control charts"""
    
    # Speed vs Time (Google Trends style)
    fig_speed = go.Figure()
    times = np.array([log.get('time', 0) for log in simulation_logs])
    speeds = np.array([log.get('speed', 0) for log in simulation_logs])
    
    fig_speed.add_trace(go.Scatter(
        x=times, y=speeds,
        mode='lines',
        name='Velocity',
        line=dict(color='#00D4FF', width=4),
        hovertemplate='<b>Time:</b> %{x:.1f} days<br><b>Speed:</b> %{y:.0f} km/h<extra></extra>'
    ))
    fig_speed.update_layout(
        title=dict(text="🚀 VELOCITY PROFILE", font=dict(size=16, color='white')),
        xaxis_title="Mission Elapsed Time (days)",
        yaxis_title="Speed (km/h)",
        template="plotly_dark",
        height=300,
        showlegend=False,
        font=dict(color='white')
    )
    
    # Fuel Gauge Style
    fig_fuel = go.Figure()
    fuels = np.array([log.get('fuel', 100000) for log in simulation_logs])
    fig_fuel.add_trace(go.Scatter(
        x=times, y=fuels,
        mode='lines',
        name='Fuel Remaining',
        line=dict(color='#FF6B6B', width=4)
    ))
    fig_fuel.update_layout(
        title=dict(text="⛽ FUEL CONSUMPTION", font=dict(size=16, color='white')),
        xaxis_title="Mission Elapsed Time (days)",
        yaxis_title="Fuel (kg)",
        template="plotly_dark",
        height=300,
        showlegend=False,
        font=dict(color='white')
    )
    
    # 3D Trajectory
    fig_3d = go.Figure()
    if len(simulation_logs) > 1:
        xs = [log.get('x', 0) for log in simulation_logs]
        ys = [log.get('y', 0) for log in simulation_logs]
        zs = [log.get('z', 0) for log in simulation_logs]
        
        fig_3d.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color='#00FF88', width=8),
            name='Trajectory',
            hovertemplate='<b>Position:</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))
    
    # Add planets
    planets = {
        'Earth': (0, 0, 0, '#4A90E2'),
        'Mars': (2.25, 0, 0, '#CD5C5C'),
        'Jupiter': (7.78, 0, 0, '#D8CA9D'),
        'Saturn': (14.27, 0, 0, '#FAD5A5'),
        'Uranus': (28.71, 0, 0, '#4FD0E7')
    }
    
    for name, (x, y, z, color) in planets.items():
        fig_3d.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=15, color=color),
            text=[name],
            textposition="middle center",
            name=name,
            hoverinfo='name'
        ))
    
    fig_3d.update_layout(
        title=dict(text="🌌 INTERPLANETARY TRAJECTORY", font=dict(size=16, color='white')),
        scene=dict(
            xaxis_title='X (10^9 km)', yaxis_title='Y (10^9 km)', zaxis_title='Z (10^9 km)',
            bgcolor='black', camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        template="plotly_dark",
        height=450,
        showlegend=True,
        font=dict(color='white')
    )
    
    return fig_speed, fig_fuel, fig_3d

def generate_rocket_svg(position, stage='idle', fuel_level=100):
    """Animated SVG rocket for live visualization"""
    flame_intensity = max(0, (100 - fuel_level) / 100 * 2)
    
    if stage == 'launch':
        flame_color = "#FF4500"
        flame_size = "rx='35' ry='30'"
    elif stage == 'travel':
        flame_color = "#00FF88"
        flame_size = "rx='15' ry='10'"
    else:
        flame_color = "#666"
        flame_size = "rx='8' ry='5'"
    
    rocket_svg = f"""
    <svg width="220" height="450" viewBox="0 0 220 450" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="rocketBody" cx="50%" cy="20%">
          <stop offset="0%" stop-color="#5A9DF2"/>
          <stop offset="60%" stop-color="#4A90E2"/>
          <stop offset="100%" stop-color="#1E3A8A"/>
        </radialGradient>
        <radialGradient id="flame" cx="50%" cy="50%">
          <stop offset="0%" stop-color="{flame_color}"/>
          <stop offset="50%" stop-color="#FFA500"/>
          <stop offset="100%" stop-color="#FF4500"/>
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge> 
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      
      <!-- Rocket Body -->
      <rect x="60" y="80" width="100" height="280" rx="50" fill="url(#rocketBody)" filter="url(#glow)" stroke="#00D4FF" stroke-width="3"/>
      
      <!-- Nose Cone -->
      <path d="M 110 60 Q 40 80 60 160 Q 160 160 180 80 Z" fill="#FFD700" stroke="#B8860B" stroke-width="4" filter="url(#glow)"/>
      
      <!-- Windows -->
      <circle cx="90" cy="140" r="12" fill="#00FFFF" opacity="0.8" stroke="#008B8B" stroke-width="2"/>
      <circle cx="130" cy="140" r="12" fill="#00FFFF" opacity="0.8" stroke="#008B8B" stroke-width="2"/>
      
      <!-- Engine Nozzle -->
      <path d="M 70 360 L 110 420