#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit dashboard application for AI Cost Optimizer.

This dashboard provides:
1. Model performance comparison
2. Cost tracking and analysis
3. Budget optimization
4. ONNX optimization results
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import sqlite3

# Add path to parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_tracking.database import (
    get_model_costs, get_model_metrics, get_onnx_metrics, 
    get_cost_summary, insert_model_cost_estimate
)

# Set page configuration
st.set_page_config(
    page_title="AI Cost Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache database connection
@st.cache_resource
def get_db_connection():
    """Get SQLite database connection."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'costs.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Cache data loading functions
@st.cache_data(ttl=300)
def load_model_costs(model_name=None, cloud_provider=None, start_date=None, end_date=None):
    """Load model costs from database."""
    return get_model_costs(model_name, cloud_provider, start_date, end_date)

@st.cache_data(ttl=300)
def load_model_metrics(model_name=None):
    """Load model metrics from database."""
    return get_model_metrics(model_name)

@st.cache_data(ttl=300)
def load_onnx_metrics(model_name=None):
    """Load ONNX optimization metrics from database."""
    return get_onnx_metrics(model_name)

@st.cache_data(ttl=300)
def load_cost_summary(group_by='model_name', start_date=None, end_date=None):
    """Load cost summary from database."""
    return get_cost_summary(group_by, start_date, end_date)

def recommend_optimal_model(model_metrics, budget):
    """
    Recommend optimal model based on budget constraints.
    
    Args:
        model_metrics (list): List of model metrics
        budget (float): Budget in USD
    
    Returns:
        dict: Recommended model metrics
    """
    # Create DataFrame from model metrics
    df = pd.DataFrame(model_metrics)
    
    # Filter out models that haven't been costed
    if 'estimated_cost' in df.columns:
        df = df[df['estimated_cost'] > 0]
        df = df[df['estimated_cost'] <= budget]
    
    # If no models are within budget, return None
    if df.empty:
        return None
    
    # Find the model with the best accuracy within budget
    if 'accuracy' in df.columns:
        return df.sort_values('accuracy', ascending=False).iloc[0].to_dict()
    
    # If no accuracy metrics, find the model with the fastest inference time
    if 'inference_time' in df.columns:
        return df.sort_values('inference_time').iloc[0].to_dict()
    
    # If no metrics, return the cheapest model
    if 'estimated_cost' in df.columns:
        return df.sort_values('estimated_cost').iloc[0].to_dict()
    
    # If no relevant metrics, return the first model
    return df.iloc[0].to_dict()

def estimate_model_cost(model_metrics, requests_per_month):
    """
    Estimate model cost based on inference time and requests per month.
    
    Args:
        model_metrics (dict): Model metrics
        requests_per_month (int): Number of requests per month
    
    Returns:
        dict: Cost estimate
    """
    # Assume AWS p3.2xlarge instance ($3.06/hour) for cost estimation
    hourly_cost = 3.06
    seconds_per_hour = 3600
    
    # Calculate GPU hours based on inference time and requests
    inference_time = model_metrics.get('inference_time', 0.05)  # Default 50ms
    gpu_hours = (inference_time * requests_per_month) / seconds_per_hour
    
    # Calculate cost
    estimated_cost = gpu_hours * hourly_cost
    
    # Create estimate
    estimate = {
        'model_name': model_metrics.get('model_name', 'unknown'),
        'requests_per_month': requests_per_month,
        'avg_request_time': inference_time,
        'estimated_cost': estimated_cost,
        'estimated_gpu_hours': gpu_hours
    }
    
    return estimate

def plot_model_metrics(metrics):
    """
    Plot model metrics comparison.
    
    Args:
        metrics (list): List of model metrics
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    df = pd.DataFrame(metrics)
    
    # Create subplots with 2x2 grid
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=("Inference Time (s)", "Model Size (MB)", 
                                      "Parameter Count (millions)", "Accuracy"))
    
    # Plot inference time
    fig.add_trace(
        go.Bar(x=df['model_name'], y=df['inference_time'], name="Inference Time"),
        row=1, col=1
    )
    
    # Plot model size
    fig.add_trace(
        go.Bar(x=df['model_name'], y=df['model_size'], name="Model Size"),
        row=1, col=2
    )
    
    # Plot parameter count
    fig.add_trace(
        go.Bar(x=df['model_name'], y=df['parameter_count'] / 1000000, name="Parameters (millions)"),
        row=2, col=1
    )
    
    # Plot accuracy
    fig.add_trace(
        go.Bar(x=df['model_name'], y=df['accuracy'], name="Accuracy"),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=700, showlegend=False)
    
    return fig

def plot_cost_vs_accuracy(metrics, cost_estimates):
    """
    Plot cost vs accuracy tradeoff.
    
    Args:
        metrics (list): List of model metrics
        cost_estimates (list): List of cost estimates
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Merge metrics and cost estimates
    metrics_df = pd.DataFrame(metrics)
    cost_df = pd.DataFrame(cost_estimates)
    
    # Create merged dataframe
    df = metrics_df.merge(cost_df, on='model_name', how='inner')
    
    # Create figure
    fig = px.scatter(
        df,
        x="estimated_cost",
        y="accuracy",
        size="parameter_count",
        color="model_name",
        hover_name="model_name",
        log_x=True,
        title="Cost vs Accuracy Tradeoff"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Estimated Monthly Cost ($)",
        yaxis_title="Accuracy",
        height=600
    )
    
    return fig

def plot_onnx_comparison(metrics):
    """
    Plot ONNX optimization comparison.
    
    Args:
        metrics (list): List of ONNX metrics
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    df = pd.DataFrame(metrics)
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for each model and framework
    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        
        # Add PyTorch bars
        fig.add_trace(go.Bar(
            name=f"{model_name} - PyTorch",
            x=[model_name],
            y=[model_df['pytorch_time'].iloc[0]],
            marker_color='blue'
        ))
        
        # Add ONNX bars
        fig.add_trace(go.Bar(
            name=f"{model_name} - ONNX",
            x=[model_name],
            y=[model_df['onnx_time'].iloc[0]],
            marker_color='green'
        ))
        
        # Add optimized ONNX bars if available
        if not pd.isna(model_df['optimized_onnx_time'].iloc[0]):
            fig.add_trace(go.Bar(
                name=f"{model_name} - Optimized ONNX",
                x=[model_name],
                y=[model_df['optimized_onnx_time'].iloc[0]],
                marker_color='red'
            ))
    
    # Update layout
    fig.update_layout(
        title="Inference Time Comparison: PyTorch vs ONNX",
        xaxis_title="Model",
        yaxis_title="Inference Time (s)",
        barmode='group',
        height=500
    )
    
    return fig

def plot_cost_over_time(costs):
    """
    Plot cost over time.
    
    Args:
        costs (list): List of cost records
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    df = pd.DataFrame(costs)
    
    # Convert dates to datetime
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    # Group by date and model
    grouped_df = df.groupby(['start_date', 'model_name']).agg({
        'cost': 'sum'
    }).reset_index()
    
    # Create figure
    fig = px.line(
        grouped_df,
        x="start_date",
        y="cost",
        color="model_name",
        title="Cost Over Time"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cost ($)",
        height=500
    )
    
    return fig

def plot_cost_by_provider(costs):
    """
    Plot cost breakdown by cloud provider.
    
    Args:
        costs (list): List of cost records
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    df = pd.DataFrame(costs)
    
    # Group by cloud provider
    grouped_df = df.groupby('cloud_provider').agg({
        'cost': 'sum'
    }).reset_index()
    
    # Create pie chart
    fig = px.pie(
        grouped_df,
        values="cost",
        names="cloud_provider",
        title="Cost Breakdown by Cloud Provider"
    )
    
    # Update layout
    fig.update_layout(height=500)
    
    return fig

def main():
    """Main function for Streamlit dashboard."""
    # Sidebar
    st.sidebar.title("AI Cost Optimizer")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["Model Comparison", "Cost Analysis", "Cost-Accuracy Simulation", "ONNX Optimization"]
    )
    
    # Load data
    model_metrics = load_model_metrics()
    model_costs = load_model_costs()
    onnx_metrics = load_onnx_metrics()
    
    # Model Comparison page
    if page == "Model Comparison":
        st.title("Model Architecture Comparison")
        
        # Display model metrics
        if model_metrics:
            st.subheader("Performance Metrics")
            
            # Plot metrics
            fig = plot_model_metrics(model_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics table
            st.subheader("Metrics Table")
            st.dataframe(pd.DataFrame(model_metrics))
        else:
            st.info("No model metrics available. Run model comparison first.")
            st.code("python models/model_comparison.py --save-to-db")
    
    # Cost Analysis page
    elif page == "Cost Analysis":
        st.title("Cost Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Date range filter
            date_range = st.date_input(
                "Select Date Range",
                value=(
                    datetime.now() - timedelta(days=30),
                    datetime.now()
                )
            )
        
        with col2:
            # Cloud provider filter
            providers = ['All'] + list(set([cost.get('cloud_provider', 'Unknown') for cost in model_costs]))
            cloud_provider = st.selectbox("Cloud Provider", providers)
        
        with col3:
            # Model filter
            models = ['All'] + list(set([cost.get('model_name', 'Unknown') for cost in model_costs]))
            model_name = st.selectbox("Model", models)
        
        # Apply filters
        filtered_costs = model_costs
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_costs = [
                cost for cost in filtered_costs 
                if cost.get('start_date') and start_date <= datetime.strptime(cost.get('start_date'), '%Y-%m-%d').date() <= end_date
            ]
        
        if cloud_provider != 'All':
            filtered_costs = [cost for cost in filtered_costs if cost.get('cloud_provider') == cloud_provider]
        
        if model_name != 'All':
            filtered_costs = [cost for cost in filtered_costs if cost.get('model_name') == model_name]
        
        # Display costs
        if filtered_costs:
            # Cost summary
            total_cost = sum(cost.get('cost', 0) for cost in filtered_costs)
            avg_cost = total_cost / len(filtered_costs) if filtered_costs else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Cost", f"${total_cost:.2f}")
            
            with col2:
                st.metric("Average Cost", f"${avg_cost:.2f}")
            
            # Cost over time
            st.subheader("Cost Over Time")
            time_fig = plot_cost_over_time(filtered_costs)
            st.plotly_chart(time_fig, use_container_width=True)
            
            # Cost by provider
            st.subheader("Cost by Cloud Provider")
            provider_fig = plot_cost_by_provider(filtered_costs)
            st.plotly_chart(provider_fig, use_container_width=True)
            
            # Cost table
            st.subheader("Cost Details")
            st.dataframe(pd.DataFrame(filtered_costs))
        else:
            st.info("No cost data available. Run cost tracking first.")
            st.code("python cost_tracking/aws_cost_tracker.py --save-to-db")
            st.code("python cost_tracking/gcp_cost_tracker.py --save-to-db")
    
    # Cost-Accuracy Simulation page
    elif page == "Cost-Accuracy Simulation":
        st.title("Cost-Accuracy Simulation")
        
        # Budget input
        st.subheader("Budget Planning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input("Monthly Budget ($)", min_value=0.0, value=500.0, step=100.0)
        
        with col2:
            requests_per_month = st.number_input(
                "Estimated Requests per Month", 
                min_value=100, 
                value=1000000, 
                step=100000,
                format="%d"
            )
        
        # Generate cost estimates
        cost_estimates = []
        
        for metric in model_metrics:
            estimate = estimate_model_cost(metric, requests_per_month)
            cost_estimates.append(estimate)
            
            # Save estimate to database
            try:
                insert_model_cost_estimate(
                    model_name=estimate['model_name'],
                    requests_per_month=estimate['requests_per_month'],
                    avg_request_time=estimate['avg_request_time'],
                    estimated_cost=estimate['estimated_cost'],
                    estimated_gpu_hours=estimate['estimated_gpu_hours']
                )
            except Exception as e:
                st.error(f"Error saving cost estimate: {str(e)}")
        
        # Display cost estimates
        if cost_estimates:
            # Cost vs accuracy plot
            st.subheader("Cost vs Accuracy Tradeoff")
            cost_accuracy_fig = plot_cost_vs_accuracy(model_metrics, cost_estimates)
            st.plotly_chart(cost_accuracy_fig, use_container_width=True)
            
            # Recommended model
            st.subheader("Recommended Model")
            
            # Create dataframe with model metrics and cost estimates
            metrics_df = pd.DataFrame(model_metrics)
            cost_df = pd.DataFrame(cost_estimates)
            df = metrics_df.merge(cost_df, on='model_name', how='inner')
            
            # Filter models within budget
            within_budget = df[df['estimated_cost'] <= budget]
            
            if not within_budget.empty:
                # Find model with best accuracy within budget
                recommended = within_budget.sort_values('accuracy', ascending=False).iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Recommended Model", recommended['model_name'])
                
                with col2:
                    st.metric("Estimated Cost", f"${recommended['estimated_cost']:.2f}")
                
                with col3:
                    st.metric("Accuracy", f"{recommended['accuracy']:.2%}")
                
                # Show recommended model details
                st.subheader("Recommended Model Details")
                st.json(recommended.to_dict())
            else:
                st.warning("No models within budget. Consider increasing your budget or reducing request volume.")
            
            # Cost estimates table
            st.subheader("All Cost Estimates")
            st.dataframe(cost_df)
        else:
            st.info("No model metrics available. Run model comparison first.")
            st.code("python models/model_comparison.py --save-to-db")
    
    # ONNX Optimization page
    elif page == "ONNX Optimization":
        st.title("ONNX Optimization Results")
        
        if onnx_metrics:
            # ONNX comparison plot
            st.subheader("Inference Time Comparison")
            onnx_fig = plot_onnx_comparison(onnx_metrics)
            st.plotly_chart(onnx_fig, use_container_width=True)
            
            # Calculate average speedup
            df = pd.DataFrame(onnx_metrics)
            avg_onnx_speedup = df['onnx_speedup'].mean()
            avg_optimized_speedup = df['optimized_onnx_speedup'].mean() if 'optimized_onnx_speedup' in df else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average ONNX Speedup", f"{avg_onnx_speedup:.2f}x")
            
            if avg_optimized_speedup:
                with col2:
                    st.metric("Average Optimized ONNX Speedup", f"{avg_optimized_speedup:.2f}x")
            
            # Display ONNX metrics table
            st.subheader("ONNX Metrics")
            st.dataframe(df)
        else:
            st.info("No ONNX optimization metrics available. Run ONNX optimization first.")
            st.code("python models/onnx_optimization.py --model-name distilbert-base-uncased --optimize --save-to-db")

# Run the app
if __name__ == "__main__":
    main()
