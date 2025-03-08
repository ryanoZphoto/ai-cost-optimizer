#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database module for AI Cost Optimizer.

This module provides functionality to:
1. Initialize the SQLite database
2. Insert and retrieve cost data
3. Insert and retrieve model metrics
"""

import os
import sqlite3
from datetime import datetime

# Add path to parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'costs.db')

def init_db():
    """
    Initialize the SQLite database with the necessary tables.
    
    Tables:
    1. model_costs - Store cost data for models
    2. model_metrics - Store performance metrics for models
    3. onnx_metrics - Store ONNX optimization metrics
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create model_costs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_costs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        version TEXT,
        cloud_provider TEXT NOT NULL,
        gpu_tpu_type TEXT,
        gpu_tpu_usage FLOAT,
        cost FLOAT NOT NULL,
        start_date TEXT,
        end_date TEXT,
        additional_info TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create model_metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        inference_time FLOAT,
        model_size FLOAT,
        parameter_count INTEGER,
        accuracy FLOAT,
        additional_metrics TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create onnx_metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS onnx_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        pytorch_time FLOAT,
        onnx_time FLOAT,
        optimized_onnx_time FLOAT,
        onnx_speedup FLOAT,
        optimized_onnx_speedup FLOAT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create cloud_credentials table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cloud_credentials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL UNIQUE,
        credentials_json TEXT,
        last_updated TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create model_cost_estimates table for budget planning
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_cost_estimates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        requests_per_month INTEGER,
        avg_request_time FLOAT,
        estimated_cost FLOAT,
        estimated_gpu_hours FLOAT,
        additional_info TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {DB_PATH}")

def insert_model_cost(model_name, cost, cloud_provider, version=None, gpu_tpu_type=None, 
                     gpu_tpu_usage=None, start_date=None, end_date=None, additional_info=None):
    """
    Insert cost data for a model into the database.
    
    Args:
        model_name (str): Name of the model
        cost (float): Cost in USD
        cloud_provider (str): Cloud provider (AWS, GCP, Azure)
        version (str, optional): Model version
        gpu_tpu_type (str, optional): GPU/TPU type used
        gpu_tpu_usage (float, optional): GPU/TPU usage in hours
        start_date (str, optional): Start date of the cost period
        end_date (str, optional): End date of the cost period
        additional_info (str, optional): Additional information as JSON string
    
    Returns:
        int: ID of the inserted record
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO model_costs (
        model_name, version, cloud_provider, gpu_tpu_type, gpu_tpu_usage, 
        cost, start_date, end_date, additional_info
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        model_name, version, cloud_provider, gpu_tpu_type, gpu_tpu_usage,
        cost, start_date, end_date, additional_info
    ))
    
    record_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return record_id

def insert_model_metrics(model_name, inference_time=None, model_size=None, 
                        parameter_count=None, accuracy=None, additional_metrics=None):
    """
    Insert performance metrics for a model into the database.
    
    Args:
        model_name (str): Name of the model
        inference_time (float, optional): Inference time in seconds
        model_size (float, optional): Model size in MB
        parameter_count (int, optional): Number of parameters
        accuracy (float, optional): Accuracy score between 0 and 1
        additional_metrics (str, optional): Additional metrics as JSON string
    
    Returns:
        int: ID of the inserted record
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO model_metrics (
        model_name, inference_time, model_size, parameter_count, accuracy, additional_metrics
    )
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        model_name, inference_time, model_size, parameter_count, accuracy, additional_metrics
    ))
    
    record_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return record_id

def insert_onnx_optimization_metrics(model_name, pytorch_time, onnx_time, optimized_onnx_time=None,
                                    onnx_speedup=None, optimized_onnx_speedup=None):
    """
    Insert ONNX optimization metrics for a model into the database.
    
    Args:
        model_name (str): Name of the model
        pytorch_time (float): PyTorch inference time in seconds
        onnx_time (float): ONNX inference time in seconds
        optimized_onnx_time (float, optional): Optimized ONNX inference time in seconds
        onnx_speedup (float, optional): Speedup factor for ONNX
        optimized_onnx_speedup (float, optional): Speedup factor for optimized ONNX
    
    Returns:
        int: ID of the inserted record
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO onnx_metrics (
        model_name, pytorch_time, onnx_time, optimized_onnx_time, onnx_speedup, optimized_onnx_speedup
    )
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        model_name, pytorch_time, onnx_time, optimized_onnx_time, onnx_speedup, optimized_onnx_speedup
    ))
    
    record_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return record_id

def insert_model_cost_estimate(model_name, requests_per_month, avg_request_time, 
                             estimated_cost, estimated_gpu_hours, additional_info=None):
    """
    Insert cost estimate for a model into the database.
    
    Args:
        model_name (str): Name of the model
        requests_per_month (int): Estimated number of requests per month
        avg_request_time (float): Average request time in seconds
        estimated_cost (float): Estimated cost in USD
        estimated_gpu_hours (float): Estimated GPU hours
        additional_info (str, optional): Additional information as JSON string
    
    Returns:
        int: ID of the inserted record
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO model_cost_estimates (
        model_name, requests_per_month, avg_request_time, estimated_cost, estimated_gpu_hours, additional_info
    )
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        model_name, requests_per_month, avg_request_time, estimated_cost, estimated_gpu_hours, additional_info
    ))
    
    record_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return record_id

def get_model_costs(model_name=None, cloud_provider=None, start_date=None, end_date=None):
    """
    Get cost data for models from the database.
    
    Args:
        model_name (str, optional): Filter by model name
        cloud_provider (str, optional): Filter by cloud provider
        start_date (str, optional): Filter by start date
        end_date (str, optional): Filter by end date
    
    Returns:
        list: List of cost records as dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    query = "SELECT * FROM model_costs WHERE 1=1"
    params = []
    
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    
    if cloud_provider:
        query += " AND cloud_provider = ?"
        params.append(cloud_provider)
    
    if start_date:
        query += " AND start_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND end_date <= ?"
        params.append(end_date)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert rows to dictionaries
    results = []
    for row in rows:
        results.append(dict(row))
    
    conn.close()
    
    return results

def get_model_metrics(model_name=None):
    """
    Get performance metrics for models from the database.
    
    Args:
        model_name (str, optional): Filter by model name
    
    Returns:
        list: List of metric records as dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    query = "SELECT * FROM model_metrics WHERE 1=1"
    params = []
    
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert rows to dictionaries
    results = []
    for row in rows:
        results.append(dict(row))
    
    conn.close()
    
    return results

def get_onnx_metrics(model_name=None):
    """
    Get ONNX optimization metrics for models from the database.
    
    Args:
        model_name (str, optional): Filter by model name
    
    Returns:
        list: List of ONNX metric records as dictionaries
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    query = "SELECT * FROM onnx_metrics WHERE 1=1"
    params = []
    
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert rows to dictionaries
    results = []
    for row in rows:
        results.append(dict(row))
    
    conn.close()
    
    return results

def get_cost_summary(group_by='model_name', start_date=None, end_date=None):
    """
    Get a summary of costs grouped by the specified field.
    
    Args:
        group_by (str): Field to group by (model_name, cloud_provider, version)
        start_date (str, optional): Filter by start date
        end_date (str, optional): Filter by end date
    
    Returns:
        list: List of summary records as dictionaries
    """
    valid_group_fields = ['model_name', 'cloud_provider', 'version', 'gpu_tpu_type']
    if group_by not in valid_group_fields:
        raise ValueError(f"Invalid group_by field. Must be one of {valid_group_fields}")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    query = f"""
    SELECT {group_by}, 
           SUM(cost) as total_cost, 
           SUM(gpu_tpu_usage) as total_gpu_tpu_usage,
           COUNT(*) as record_count,
           MIN(start_date) as min_date,
           MAX(end_date) as max_date
    FROM model_costs
    WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND start_date >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND end_date <= ?"
        params.append(end_date)
    
    query += f" GROUP BY {group_by} ORDER BY total_cost DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Convert rows to dictionaries
    results = []
    for row in rows:
        results.append(dict(row))
    
    conn.close()
    
    return results

def main():
    """Initialize the database when run directly."""
    print("Initializing AI Cost Optimizer database...")
    init_db()
    
    print("Adding sample data...")
    
    # Add sample model costs
    insert_model_cost(
        model_name="bert-base-uncased",
        version="1.0",
        cloud_provider="AWS",
        gpu_tpu_type="p3.2xlarge",
        gpu_tpu_usage=10.5,
        cost=42.0,
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    insert_model_cost(
        model_name="distilbert-base-uncased",
        version="1.0",
        cloud_provider="AWS",
        gpu_tpu_type="p3.2xlarge",
        gpu_tpu_usage=8.2,
        cost=32.8,
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    insert_model_cost(
        model_name="bert-tiny",
        version="1.0",
        cloud_provider="GCP",
        gpu_tpu_type="n1-standard-8",
        gpu_tpu_usage=5.0,
        cost=15.0,
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    # Add sample model metrics
    insert_model_metrics(
        model_name="bert-base-uncased",
        inference_time=0.052,
        model_size=440.0,
        parameter_count=110000000,
        accuracy=0.92
    )
    
    insert_model_metrics(
        model_name="distilbert-base-uncased",
        inference_time=0.031,
        model_size=265.0,
        parameter_count=66000000,
        accuracy=0.89
    )
    
    insert_model_metrics(
        model_name="bert-tiny",
        inference_time=0.008,
        model_size=17.5,
        parameter_count=4400000,
        accuracy=0.78
    )
    
    # Add sample ONNX metrics
    insert_onnx_optimization_metrics(
        model_name="bert-base-uncased",
        pytorch_time=0.052,
        onnx_time=0.023,
        optimized_onnx_time=0.017,
        onnx_speedup=2.26,
        optimized_onnx_speedup=3.06
    )
    
    insert_onnx_optimization_metrics(
        model_name="distilbert-base-uncased",
        pytorch_time=0.031,
        onnx_time=0.014,
        optimized_onnx_time=0.011,
        onnx_speedup=2.21,
        optimized_onnx_speedup=2.82
    )
    
    # Add sample cost estimates
    insert_model_cost_estimate(
        model_name="bert-base-uncased",
        requests_per_month=1000000,
        avg_request_time=0.052,
        estimated_cost=520.0,
        estimated_gpu_hours=14.4
    )
    
    insert_model_cost_estimate(
        model_name="distilbert-base-uncased",
        requests_per_month=1000000,
        avg_request_time=0.031,
        estimated_cost=310.0,
        estimated_gpu_hours=8.6
    )
    
    insert_model_cost_estimate(
        model_name="bert-tiny",
        requests_per_month=1000000,
        avg_request_time=0.008,
        estimated_cost=80.0,
        estimated_gpu_hours=2.2
    )
    
    print("Database initialized with sample data.")

if __name__ == "__main__":
    main() 