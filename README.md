# SentraVigil-Intelligent-Safety-Decision-Support-Engine

SentraVigil is an enterprise-grade, zero-latency autonomous threat detection and situational awareness platform built for coastal defense, restricted zone monitoring, and tourist safety. It combines on-device deep learning (GRU-VAE behavioral anomaly detection), simulated LTE edge-to-command broadcasting, Gemini LLM tactical decision support, and a dual-service Node.js / Python architecture.

---

## 🏗️ System Architecture & Tech Stack

```
[Tourist Handset / Edge Device]
          │ (Inertial & GPS Sensor Stream)
          ▼
[Edge GRU-VAE (ONNX Runtime, trained checkpoint)] ──(Anomaly > 0.45 Threshold)──┐
          │ (Normal Telemetry)                                     │
          ▼                                                        ▼
   [Central Command Map] <──(Simulated LTE Uplink)────── [Encrypted LTE Gateway]
          │
          ├─► [Gemini AI Neural Reasoning Engine]
          └─► [One-Click Police Dispatcher & TTS Voice Alert]
```

- **Backend Gateway**: Node.js & Express managing dual-port service orchestration.
- **ML / Edge Inference Core**: Python ONNX Runtime inference against the trained GRU-VAE checkpoint, communicating through correlated JSON lines over standard I/O. A documented physical-threshold safety net supplements the learned score for restricted-zone and severe telemetry cases.
- **AI Decision Support**: Gemini 2.0 Flash REST integration with a robust tactical fallback matrix.
- **Command Frontend (Port 8000)**: Modern dark-themed government control room featuring Leaflet.js interactive maps, draggable resizable panels, and real-time telemetry streaming.
- **Mobile Handset Simulator (Port 8001)**: Standalone iPhone-style mobile app simulating on-device edge computing, haptic feedback, custom anomaly injection, and persistent location privacy.

---
## Model Evaluation

Sentravigil’s anomaly detector is a GRU-VAE trained on simulated tourist-mobility telemetry. It evaluates rolling windows of 20 timesteps across six behavioural features:

- Average speed
- Inactivity duration
- Route deviation
- Direction changes
- Restricted-zone level
- Journey duration

The model flags an anomaly when reconstruction error exceeds the calibrated threshold of **0.45**.

### Evaluation Results

| Metric | Mean Score | Standard Deviation |
|---|---:|---:|
| Precision | 0.8795 | 0.0600 |
| Recall | 0.4523 | ~0.0000 |
| F1 Score | 0.5974 | 0.0000 |
| Accuracy | 0.9079 | 0.0100 |
| Test AUC-ROC | 0.9869 | ~0.0000 |

### Interpretation

The model achieves strong precision and AUC-ROC, meaning that when it flags an incident, it is usually correct and it separates normal from anomalous behaviour effectively. Recall is lower at the selected operating threshold, which reflects a conservative alerting strategy intended to reduce false alarms.

For high-severity physical-risk conditions—such as restricted-zone entry, extreme route deviation, or prolonged inactivity—the live system also applies an explicit safety-net policy in addition to the learned GRU-VAE score.

## 🚀 Key Features

### 1. On-Device Edge Anomaly Detection (GRU-VAE)
- Continuously samples 6 spatial-temporal features: `avg_speed`, `inactivity_duration`, `route_deviation`, `direction_changes`, `zone_level`, and `journey_duration`.
- Uses sliding windows (20 timesteps) to calculate reconstruction error against a calibrated 99th-percentile baseline ($Threshold = 0.45$).
- Instantly detects cliff falls, physical incapacitation, restricted zone intrusion, and panic flight.

### 2. Dual-Port Enterprise Dashboard
- **Central Intelligence Center (`http://127.0.0.1:8000`)**: Real-time government command room. Features draggable sidebar and panel resizers, Leaflet map layers (Streets & Satellite), and live anomaly tracking.
- **Mobile Edge Simulator (`http://127.0.0.1:8001`)**: Smartphone interface displaying live sensor telemetry, edge protection status, and physical vibration haptic feedback.

### 3. Persistent Location Secrecy (Privacy Policy)
- Enterprise-grade location masking enforced at the **server data-access layer** before coordinates reach Central Command.
- When enabled, exact coordinates are transformed into a privacy-preserving ~500m grid cell (`APPROXIMATE_GEOCELL`).
- Persists across page reloads and browser restarts via `privacy_db.json`.
- Restricts operators from assigning custom routes to private tourists.

### 4. Interactive Map Drawing Tools
- **Draw Restricted Geofence**: Draw custom danger polygons directly on the map. Tourists entering the zone trigger immediate high-risk alerts.
- **Draw Custom Route**: Draw polylines to assign custom navigation paths for selected tourists.

### 5. Gemini AI Tactical Decision Support & TTS Voice
- When an anomaly fires, Gemini AI analyzes feature contributions and provides structured tactical assessments, threat classifications, and 3-step Standard Operating Procedures (SOPs).
- **Text-to-Speech (TTS)**: Clicking "Dispatch Tactical Patrol" synthesizes a computerized control-room voice announcing the rescue mission.

### 6. Bidirectional Communication & Police Dispatch
- **Command ➔ Handset**: Push security check-ins that trigger physical haptic vibrations and red screen flashes on the mobile app.
- **One-Click Dispatch**: Instantly dispatches local police tactical units with real-time ETA tracking and logs them to the shift history.

---

## 🛠️ Installation & Setup

1. **Install Node.js Dependencies**:
   ```bash
   npm install
   ```

2. **Configure Gemini API Key (Optional)**:
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your_api_key_here"
   ```
   *(Note: If no API key is provided, Sentravigil automatically switches to its built-in tactical fallback analytical engine, ensuring uninterrupted demo performance).*

3. **Configure the prototype API token (recommended)**:
   ```bash
   # Windows PowerShell
   $env:Sentravigil_API_TOKEN="choose-a-demo-token"
   ```
   Mutating API routes require this bearer token. The dashboards receive it through their server-rendered meta tag for prototype use; do not use the fallback token outside local demos.

4. **Launch the Enterprise Server**:
   ```bash
   node server.js
   ```

---

## 🖥️ Accessing the Prototype

Open two browser tabs/windows:
- **Central Intelligence Center**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Mobile Handset Simulator**: [http://127.0.0.1:8001](http://127.0.0.1:8001)

---

## 🎬 Recommended Demo Flow for Judges

1. **Tourists Overview**: Show the live telemetry stream on the Central Command sidebar and switch map layers between **Streets** and **Satellite** view.
2. **Path Control**: Select a tourist (e.g., TS-101) and use the map tool to draw a custom route. Watch them lock onto and follow your trajectory.
3. **Geofencing**: Draw a restricted geofence polygon across a tourist's path. Witness the automatic reconstruction error spike and LTE warning flash.
4. **Mobile Edge Simulator**: Open the **Custom Anomaly Studio** on the mobile app, slide Inactivity to 150s, and inject a custom vector.
5. **AI Reasoning**: Review the Gemini AI threat analysis card on Central Command, click **Dispatch**, and listen to the computerized TTS dispatch voice.
6. **Location Secrecy**: Click **Location Secrecy: ON** on the mobile app. Observe the marker instantly transform into an approximate ~500m geocell on the map. Reload the page to prove persistence.

## Limitations and Next Steps

This prototype was trained and evaluated on simulated telemetry because real tourist incident data is sensitive and difficult to obtain. Future work includes evaluation on ethically sourced real-world data, production identity management, stronger secret handling, and cloud deployment.
