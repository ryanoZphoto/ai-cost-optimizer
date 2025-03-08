#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock data generator for AWS and GCP cost tracking.

This module provides functions to generate realistic mock data for AWS and GCP costs,
allowing users to demonstrate the cost tracking functionality without actual cloud credentials.
"""

import random
import datetime
import json
from datetime import datetime, timedelta

def generate_mock_aws_costs(start_date=None, end_date=None, model_name=None):
    """
    Generate mock AWS costs data.
    
    Args:
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        model_name (str, optional): Model name to filter by
    
    Returns:
        dict: Mock AWS Cost Explorer response
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert to datetime for iteration
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Define model costs per day (in USD)
    model_costs = {
        "bert-base-uncased": {"p3.2xlarge": (3.06, 8)},  # (hourly cost, hours)
        "distilbert-base-uncased": {"p3.2xlarge": (3.06, 5)},
        "prajjwal1/bert-tiny": {"p3.2xlarge": (3.06, 2)},
        "gpt2": {"p3.8xlarge": (12.24, 10)},
        "t5-small": {"p3.2xlarge": (3.06, 4)}
    }
    
    # Filter by model name if provided
    if model_name and model_name in model_costs:
        filtered_models = {model_name: model_costs[model_name]}
    else:
        filtered_models = model_costs
    
    # Generate results
    results_by_time = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        groups = []
        
        for model, instances in filtered_models.items():
            for instance_type, (hourly_cost, hours) in instances.items():
                # Add some randomness
                daily_hours = hours * (0.9 + random.random() * 0.2)  # +/- 10%
                daily_cost = hourly_cost * daily_hours
                
                groups.append({
                    "Keys": [f"Amazon EC2 - {instance_type} - {model}"],
                    "Metrics": {
                        "BlendedCost": {
                            "Amount": str(round(daily_cost, 2)),
                            "Unit": "USD"
                        },
                        "UsageQuantity": {
                            "Amount": str(round(daily_hours, 2)),
                            "Unit": "Hours"
                        }
                    }
                })
        
        results_by_time.append({
            "TimePeriod": {
                "Start": current_dt.strftime('%Y-%m-%d'),
                "End": (current_dt + timedelta(days=1)).strftime('%Y-%m-%d')
            },
            "Groups": groups
        })
        
        current_dt += timedelta(days=1)
    
    return {
        "ResultsByTime": results_by_time
    }

def generate_mock_gcp_costs(billing_account, start_date=None, end_date=None, model_name=None):
    """
    Generate mock GCP costs data.
    
    Args:
        billing_account (str): GCP billing account ID
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        model_name (str, optional): Model name to filter by
    
    Returns:
        list: List of mock GCP cost records
    """
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Define model costs
    model_costs = {
        "bert-base-uncased": {"n1-standard-8-t4": 35.0},
        "distilbert-base-uncased": {"n1-standard-8-t4": 22.0},
        "prajjwal1/bert-tiny": {"n1-standard-4": 12.0},
        "gpt2": {"a2-highgpu-1g": 55.0},
        "t5-small": {"n1-standard-8-t4": 28.0}
    }
    
    # Filter by model name if provided
    if model_name and model_name in model_costs:
        filtered_models = {model_name: model_costs[model_name]}
    else:
        filtered_models = model_costs
    
    results = []
    
    for model, instances in filtered_models.items():
        for machine_type, cost in instances.items():
            # Add some randomness
            adjusted_cost = cost * (0.9 + random.random() * 0.2)  # +/- 10%
            
            # Determine if it's a GPU or TPU
            if 't4' in machine_type or 'highgpu' in machine_type:
                gpu_tpu_type = machine_type
            else:
                gpu_tpu_type = None
            
            record = {
                'model_name': model,
                'version': '1.0',
                'cloud_provider': 'GCP',
                'service_name': 'Compute Engine',
                'gpu_tpu_type': gpu_tpu_type,
                'gpu_tpu_usage': random.uniform(5.0, 15.0),  # Random hours
                'cost': round(adjusted_cost, 2),
                'start_date': start_date,
                'end_date': end_date,
                'project_id': 'mock-project-123',
                'resource_type': 'compute.googleapis.com/Instance'
            }
            
            results.append(record)
    
    return results

# Patch the AWS Cost Tracker to use mock data when credentials are missing
def patch_aws_cost_tracker():
    """
    Monkey patch the AWSCostTracker to use mock data when AWS credentials are missing.
    """
    from . import aws_cost_tracker
    original_get_costs = aws_cost_tracker.AWSCostTracker.get_costs
    
    def patched_get_costs(self, *args, **kwargs):
        try:
            return original_get_costs(self, *args, **kwargs)
        except Exception as e:
            if "Unable to locate credentials" in str(e):
                print("AWS credentials not found. Using mock data.")
                return generate_mock_aws_costs(
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    model_name=kwargs.get('model_name')
                )
            else:
                raise
    
    # Apply the patch
    aws_cost_tracker.AWSCostTracker.get_costs = patched_get_costs

# Patch the GCP Cost Tracker to use mock data
def patch_gcp_cost_tracker():
    """
    Monkey patch the GCPCostTracker to use mock data when GCP credentials are missing.
    """
    try:
        from . import gcp_cost_tracker
        original_get_costs = gcp_cost_tracker.GCPCostTracker.get_costs
        
        def patched_get_costs(self, billing_account, *args, **kwargs):
            try:
                return original_get_costs(self, billing_account, *args, **kwargs)
            except Exception as e:
                print("GCP credentials not found or billing account invalid. Using mock data.")
                return generate_mock_gcp_costs(
                    billing_account=billing_account,
                    start_date=kwargs.get('start_date'),
                    end_date=kwargs.get('end_date'),
                    model_name=kwargs.get('model_name')
                )
        
        # Apply the patch
        gcp_cost_tracker.GCPCostTracker.get_costs = patched_get_costs
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not patch GCP Cost Tracker: {str(e)}")

def apply_patches():
    """
    Apply all patches to use mock data.
    """
    patch_aws_cost_tracker()
    patch_gcp_cost_tracker()

if __name__ == "__main__":
    # Demo
    aws_costs = generate_mock_aws_costs()
    gcp_costs = generate_mock_gcp_costs("mock-billing-account")
    
    print("AWS Costs Sample:")
    print(json.dumps(aws_costs["ResultsByTime"][0], indent=2))
    
    print("\nGCP Costs Sample:")
    print(json.dumps(gcp_costs[0], indent=2)) 