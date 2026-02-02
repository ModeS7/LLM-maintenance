# Offshore Vessel Monitoring System

An AI-powered monitoring system for offshore vessel power data with anomaly detection and natural language querying.

## Features

- **Transformer Autoencoder** for anomaly detection using reconstruction error
- **Real-Time Dashboard** showing vessel variables by category
- **LLM Chat Interface** with tool calling for natural language queries
- **Charts Page** for detailed variable analysis and anomaly visualization

## Data

The system uses power monitoring data from M/S Olympic Hera (Offshore Construction Vessel):
- ~1.58M rows covering 91 days of operation
- 5-second sampling rate
- 16 monitored variables across electrical, maneuver, propulsion, and navigation systems

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For LLM features, install Ollama: https://ollama.ai
# Then pull a model:
ollama pull qwen3:8b
```

## Usage

### 1. Train the Model

```bash
python -m src.train
```

This will:
- Load and preprocess the vessel data
- Train the Transformer Autoencoder
- Save the model to `models/autoencoder.pt`
- Save the scaler to `models/scaler.pkl`

Training options:
```bash
python -m src.train --epochs 50 --batch-size 32 --lr 1e-4
```

### 2. Run the Application

```bash
python -m src.app
```

Then open http://localhost:7860 in your browser.

Options:
```bash
python -m src.app --host 0.0.0.0 --port 8080 --share
```

## Project Structure

```
├── data/
│   └── Data_Pwr_All_S5.txt      # Vessel power data
├── models/
│   ├── autoencoder.pt           # Trained model
│   └── scaler.pkl               # Feature scaler
├── src/
│   ├── __init__.py
│   ├── app.py                   # Gradio UI
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── model.py                 # Transformer Autoencoder
│   ├── train.py                 # Training script
│   ├── inference.py             # Anomaly detection
│   ├── llm_agent.py             # LLM with tool calling
│   ├── tools.py                 # Tool definitions
│   └── visualization.py         # Chart utilities
├── requirements.txt
└── README.md
```

## Variable Groups

| Group | Variables |
|-------|-----------|
| Electrical | Bus1_Load, Bus1_Avail_Load, Bus2_Load, Bus2_Avail_Load |
| Maneuver | BowThr1_Power, BowThr2_Power, BowThr3_Power, SternThr1_Power, SternThr2_Power |
| Propulsion | Main_Prop_PS_Drive_Power, Main_Prop_SB_Drive_Power, Main_Prop_PS_ME1_Power, Main_Prop_PS_ME2_Power |
| Ship | Draft_Aft, Draft_Fwd, Speed |
| Coordinates | Latitude, Longitude |

## Model Architecture

The Transformer Autoencoder uses:
- Input dimension: 16 features
- Embedding dimension: 64
- Attention heads: 4
- Encoder/Decoder layers: 2 each
- Window size: 120 timesteps
- Anomaly detection via reconstruction error

## LLM Tools

The chat interface supports these tools:
- `get_vessel_status`: Current operational status
- `get_variable_readings`: Readings for a variable group
- `get_anomaly_history`: Recent anomaly events
- `get_variable_chart_data`: Time series data for plotting
- `analyze_anomaly`: Detailed anomaly analysis