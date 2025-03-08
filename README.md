# AI Cost Optimization Tool

A comprehensive Python-based tool for optimizing AI model costs, comparing model architectures, tracking cloud costs, and simulating cost-accuracy tradeoffs.

## Features

1. **Model Architecture Recommendations**:
   - Compare current models (e.g., BERT) with lightweight alternatives (e.g., DistilBERT, TinyBERT)
   - Use ONNX runtime to optimize model inference speed

2. **Cost Tracking**:
   - Integrate AWS/GCP billing APIs to fetch GPU/TPU usage costs
   - Track costs per model version in a SQLite database

3. **Cost-Accuracy Simulation**:
   - Interactive dashboard (Streamlit) showing tradeoffs between model accuracy and inference costs
   - Input budget constraints and get recommended optimal models

## Project Structure

```
ai-cost-optimizer/
├── models/
│   ├── model_comparison.py  - Compare different model architectures
│   └── onnx_optimization.py - Convert and optimize models with ONNX
├── cost_tracking/
│   ├── aws_cost_tracker.py  - Track AWS GPU costs
│   ├── gcp_cost_tracker.py  - Track GCP TPU costs
│   └── database.py          - SQLite database operations
├── dashboard/
│   └── app.py               - Streamlit dashboard application
├── data/
│   └── costs.db             - SQLite database (created on first run)
├── requirements.txt         - Project dependencies
└── README.md                - This file
```

## Setup Instructions

1. **Install dependencies**:
   ```
   cd ai-cost-optimizer
   pip install -r requirements.txt
   ```

2. **Configure cloud credentials**:
   - For AWS: Configure AWS credentials using `aws configure` or environment variables
   - For GCP: Set up application default credentials using `gcloud auth application-default login`

3. **Initialize the database**:
   ```
   python cost_tracking/database.py
   ```

4. **Run the model comparison**:
   ```
   python models/model_comparison.py
   ```

5. **Run the dashboard**:
   ```
   cd dashboard
   streamlit run app.py
   ```

## Usage Examples

### Model Comparison
Compare different models' performance:
```
python models/model_comparison.py --models bert-base-uncased,distilbert-base-uncased,prajjwal1/bert-tiny
```

### Cost Tracking
Track AWS costs for a specific period:
```
python cost_tracking/aws_cost_tracker.py --start-date 2024-01-01 --end-date 2024-01-31
```

### ONNX Optimization
Convert a model to ONNX format:
```
python models/onnx_optimization.py --model-name distilbert-base-uncased --output distilbert.onnx
```

## Requirements

- Python 3.8+
- AWS/GCP accounts (for cost tracking features)
- GPU/TPU (optional, for performance benchmarking)

## License

MIT 