# Offshore Vessel Monitoring System with LLM Interface

A proof-of-concept showing how conversational AI makes vessel monitoring accessible. Uses power data from M/S Olympic Hera, a Transformer Autoencoder for anomaly detection, and a local LLM (Ollama) for natural language interaction.

## Key Idea

**LLM interprets, model detects.** The LLM never makes up numbers - it calls tools to get real anomaly detection from the Transformer model, then explains them in plain language.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Gradio UI (app.py)                          │
│   Real-Time Dashboard │ Charts │ Chat + Quick Prompts           │
│   Chat History (browser local storage)                          │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 VesselAgent (llm_agent.py)                      │
│   Local LLM (Ollama) - decides which tools to call              │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Tools (tools.py)                           │
│   get_vessel_status │ get_variable_readings │ get_anomaly_history│
│   get_variable_chart_data │ analyze_anomaly │ get_trend_prediction│
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AnomalyDetector (inference.py)                │
│   Transformer Autoencoder (models/autoencoder.pt) → Anomalies   │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 VesselDataLoader (data_loader.py)               │
│   M/S Olympic Hera dataset (~1.58M rows, 91 days)               │
└─────────────────────────────────────────────────────────────────┘
```

## How the LLM Gets Data

The LLM **never sees raw data**. It uses tool-calling:

```
User: "What is the vessel status?"
  → LLM decides to call get_vessel_status()
  → Tool returns JSON: {status: "Normal", anomaly_score: 0.12, ...}
  → LLM interprets: "The vessel is operating normally with low anomaly score..."
```

**Why tool-calling?**
- Raw data → hallucinations
- Fine-tuning → outdated when data changes
- Tool-calling → real-time data, LLM interprets

## Tutorial: Getting Everything Working

### Prerequisites

- Python 3.10+
- 5-6GB disk space (for Ollama model)

### Step 1: Clone and Setup

**Linux / macOS / WSL:**
```bash
git clone <repository-url>
cd LLM-maintenance

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
git clone <repository-url>
cd LLM-maintenance

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
git clone <repository-url>
cd LLM-maintenance

python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Step 2: Install Ollama (Local LLM)

| Platform | Installation |
|----------|--------------|
| **Windows** | Download installer from https://ollama.ai/download |
| **macOS** | `brew install ollama` or download from https://ollama.ai/download |
| **Linux/WSL** | `curl -fsSL https://ollama.ai/install.sh \| sh` |

After installation, pull the model (same command on all platforms):
```
ollama pull qwen3:8b
```

**Verify:** `ollama list` should show `qwen3:8b`.

> **Note (Windows):** Ollama runs automatically after installation. If you see "connection refused" errors, open the Ollama app from the Start menu.

### Step 3: Train the Model

```bash
python -m src.train
```

This trains the Transformer Autoencoder and saves to `models/autoencoder.pt`.

**Options:**
```bash
python -m src.train --epochs 50 --batch-size 512 --lr 1e-4
```

### Step 4: Run the Demo

```
python -m src.app
```

**Open:** http://localhost:7860

**Options:**
```bash
python -m src.app --host 0.0.0.0 --port 8080 --share
```

### Step 5: Try It Out

1. **Real-Time Dashboard** - View live vessel data by category
2. **Click quick prompt buttons** - "Vessel status", "Electrical readings", etc.
3. **Ask custom questions** - "Are there any anomalies?" or "Show propulsion power"
4. **Charts page** - Select variables for detailed analysis
5. **Chat history** - Previous conversations saved in left sidebar

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Ollama connection refused` | Run `ollama serve` in another terminal |
| `Model not found` | Run `ollama pull qwen3:8b` |
| Chat history not showing | Refresh page, history is in browser local storage |

## Dataset

**M/S Olympic Hera** - Offshore Construction Vessel power monitoring data.

- ~1.58M rows covering 91 days of operation
- 5-second sampling rate
- 16 monitored variables

### Variable Groups

| Group | Variables |
|-------|-----------|
| Electrical | Bus1_Load, Bus1_Avail_Load, Bus2_Load, Bus2_Avail_Load |
| Maneuver | BowThr1_Power, BowThr2_Power, BowThr3_Power, SternThr1_Power, SternThr2_Power |
| Propulsion | Main_Prop_PS_Drive_Power, Main_Prop_SB_Drive_Power, Main_Prop_PS_ME1_Power, Main_Prop_PS_ME2_Power |
| Ship | Draft_Aft, Draft_Fwd, Speed |
| Coordinates | Latitude, Longitude |

## Model Architecture

**Transformer Autoencoder** - Detects anomalies via reconstruction error.

| Parameter | Value |
|-----------|-------|
| Input dimension | 16 features |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Window size | 120 timesteps |

## LLM Tools

| Tool | Description |
|------|-------------|
| `get_vessel_status` | Current operational status and anomaly score |
| `get_variable_readings` | Readings for a variable group |
| `get_anomaly_history` | Recent anomaly events |
| `get_variable_chart_data` | Time series data for plotting |
| `analyze_anomaly` | Detailed anomaly analysis |
| `get_trend_prediction` | Trend analysis and time-to-failure prediction |

## Project Structure

```
├── run_cbm_evaluation.py  # CBM evaluation pipeline (fault injection + detection)
├── requirements.txt
├── models/
│   ├── autoencoder.pt     # Trained Transformer Autoencoder checkpoint
│   └── scaler.pkl         # Fitted StandardScaler for feature normalization
├── src/
│   ├── app.py             # Gradio UI
│   ├── llm_agent.py       # Ollama LLM with tool calling
│   ├── tools.py           # 6 tools LLM can call
│   ├── inference.py       # Transformer model loading & anomaly detection
│   ├── model.py           # Transformer Autoencoder architecture
│   ├── train.py           # Training script
│   ├── cbm.py             # Condition-based maintenance pipeline
│   ├── data_loader.py     # Vessel data parsing
│   └── visualization.py   # Chart utilities
└── static/                # UI assets
```

## Anomaly Severity

| Level | Threshold | Action |
|-------|-----------|--------|
| Critical | > 0.8 | Immediate investigation |
| Warning | 0.5 - 0.8 | Schedule inspection |
| Normal | < 0.5 | Normal operation |
