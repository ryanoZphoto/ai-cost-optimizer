import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the dashboard app - handle import errors gracefully
try:
    # Initialize database if it doesn't exist
    from cost_tracking.database import init_db
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'costs.db')
    if not os.path.exists(db_path):
        print("Initializing database...")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        init_db()
        
    # Now import the main dashboard app
    from dashboard.app import main
    
    # Set a flag to show we're in demo mode with limited functionality
    os.environ["AI_COST_OPTIMIZER_DEMO_MODE"] = "1"
    
    # Run the dashboard
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    # If imports fail, provide a simplified dashboard
    import streamlit as st
    
    st.set_page_config(page_title="AI Cost Optimizer", page_icon="💰")
    
    st.title("AI Cost Optimizer - Demo Mode")
    
    st.warning("""
    Some dependencies could not be loaded on this Streamlit Cloud environment.
    
    This is a demonstration version with limited functionality.
    For the full experience with ONNX optimization and cloud provider integration,
    please run the application locally by following these steps:
    
    1. Clone the repository: `git clone https://github.com/ryanoZphoto/ai-cost-optimizer.git`
    2. Install requirements: `pip install -r requirements.txt`
    3. Initialize the database: `python main.py init`
    4. Run the dashboard: `python main.py dashboard`
    """)
    
    st.subheader("About the AI Cost Optimizer")
    
    st.markdown("""
    The AI Cost Optimizer helps you:
    
    1. **Compare Model Architectures**:
       - See performance differences between BERT, DistilBERT, and BERT-tiny
       - Measure inference time, model size, parameter count, and accuracy
       
    2. **Track Cloud Costs**:
       - Monitor AWS and GCP GPU/TPU usage
       - Track costs per model version
       
    3. **Simulate Cost-Accuracy Tradeoffs**:
       - Find the optimal model for your budget
       - Balance performance and cost
       
    4. **Optimize with ONNX**:
       - Convert models to ONNX format
       - Achieve 2-3x speedup in inference time
    """)
    
    st.image("https://miro.medium.com/max/1400/1*L76A5gL6176UbMgn7q4Ybg.png", 
             caption="Example of model accuracy vs. cost tradeoff")
    
    st.subheader("Sample Data")
    
    import pandas as pd
    import numpy as np
    
    # Create sample data for demonstration
    models = ["BERT-base", "DistilBERT", "BERT-tiny"]
    inference_time = [0.052, 0.031, 0.008]
    model_size = [440, 265, 17.5]
    accuracy = [0.92, 0.89, 0.78]
    
    df = pd.DataFrame({
        "Model": models,
        "Inference Time (s)": inference_time,
        "Model Size (MB)": model_size,
        "Accuracy": accuracy
    })
    
    st.dataframe(df)
    
    # Show a simple chart
    import plotly.express as px
    
    fig = px.bar(df, x="Model", y="Inference Time (s)")
    st.plotly_chart(fig)
    
    # Show another chart
    fig2 = px.scatter(df, x="Inference Time (s)", y="Accuracy", size="Model Size (MB)", 
                     hover_name="Model", size_max=60)
    st.plotly_chart(fig2) 