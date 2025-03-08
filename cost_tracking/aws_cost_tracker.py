#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AWS Cost Tracker module for AI Cost Optimizer.

This module provides functionality to:
1. Fetch GPU/TPU usage costs from AWS
2. Filter costs by date, service, and tags
3. Store cost data in the SQLite database
"""

import os
import argparse
import json
import boto3
from datetime import datetime, timedelta
import pandas as pd

# Add path to parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_tracking.database import insert_model_cost

# Import for mock data (will be used if credentials aren't available)
try:
    from cost_tracking.mock_data import generate_mock_aws_costs
except ImportError:
    generate_mock_aws_costs = None

class AWSCostTracker:
    """Class to track AWS GPU/TPU usage costs."""
    
    def __init__(self, profile=None, region=None):
        """
        Initialize AWS Cost Tracker.
        
        Args:
            profile (str, optional): AWS profile name
            region (str, optional): AWS region
        """
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        session = boto3.Session(**session_kwargs)
        
        self.ce_client = session.client('ce', region_name=region or 'us-east-1')
        self.current_date = datetime.now()
    
    def get_costs(self, start_date=None, end_date=None, granularity='DAILY', 
                 model_name=None, gpu_filters=None, tags=None):
        """
        Get AWS costs for GPU instances or specific ML services.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            granularity (str, optional): Time granularity (DAILY, MONTHLY)
            model_name (str, optional): Model name to filter by tags
            gpu_filters (list, optional): List of GPU instance types to filter
            tags (dict, optional): Dictionary of tags to filter
        
        Returns:
            dict: Cost data
        """
        # Set default dates if not provided
        if not start_date:
            start_date = (self.current_date - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = self.current_date.strftime('%Y-%m-%d')
        
        # Define filters
        filters = {
            'Dimensions': {
                'Key': 'SERVICE',
                'Values': [
                    'Amazon Elastic Compute Cloud - Compute',
                    'Amazon SageMaker'
                ]
            }
        }
        
        # Add GPU instance filter if provided
        if gpu_filters:
            filters['Dimensions']['Key'] = 'INSTANCE_TYPE'
            filters['Dimensions']['Values'] = gpu_filters
        
        # Add tag filters if provided
        if tags:
            tag_filters = []
            for key, value in tags.items():
                tag_filters.append({
                    'Key': f'tag:{key}',
                    'Values': [value]
                })
            
            # Add model name tag if provided
            if model_name:
                tag_filters.append({
                    'Key': 'tag:ModelName',
                    'Values': [model_name]
                })
                
            if tag_filters:
                filters['Tags'] = {
                    'Key': 'TagKey',
                    'Values': [tag for tag in tags.keys()]
                }
        
        # Make API request
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity=granularity,
            Metrics=['BlendedCost', 'UsageQuantity'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ],
            Filter=filters
        )
        
        return response
    
    def get_ec2_gpu_costs(self, start_date=None, end_date=None, granularity='DAILY'):
        """
        Get EC2 GPU instance costs.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            granularity (str, optional): Time granularity (DAILY, MONTHLY)
        
        Returns:
            dict: Cost data
        """
        gpu_instance_types = [
            'p2', 'p3', 'p4', 'p5',  # P-series (general purpose GPU)
            'g3', 'g4', 'g5',        # G-series (graphics-intensive)
            'inf1',                   # Inferentia
        ]
        
        filters = []
        for gpu_type in gpu_instance_types:
            filters.append(f'{gpu_type}.*')
        
        return self.get_costs(
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            gpu_filters=filters
        )
    
    def get_sagemaker_costs(self, start_date=None, end_date=None, granularity='DAILY', model_name=None):
        """
        Get SageMaker costs.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            granularity (str, optional): Time granularity (DAILY, MONTHLY)
            model_name (str, optional): Model name to filter by tags
        
        Returns:
            dict: Cost data
        """
        filters = {
            'Dimensions': {
                'Key': 'SERVICE',
                'Values': ['Amazon SageMaker']
            }
        }
        
        # Add model name tag if provided
        if model_name:
            filters['Tags'] = {
                'Key': 'tag:ModelName',
                'Values': [model_name]
            }
        
        # Make API request
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date or (self.current_date - timedelta(days=30)).strftime('%Y-%m-%d'),
                'End': end_date or self.current_date.strftime('%Y-%m-%d')
            },
            Granularity=granularity,
            Metrics=['BlendedCost', 'UsageQuantity'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'USAGE_TYPE'
                }
            ],
            Filter=filters
        )
        
        return response
    
    def parse_cost_response(self, response, model_name=None, version=None):
        """
        Parse AWS Cost Explorer response.
        
        Args:
            response (dict): AWS Cost Explorer response
            model_name (str, optional): Model name
            version (str, optional): Model version
        
        Returns:
            list: List of cost records
        """
        results = []
        
        for result_by_time in response.get('ResultsByTime', []):
            time_period = result_by_time.get('TimePeriod', {})
            start_date = time_period.get('Start')
            end_date = time_period.get('End')
            
            for group in result_by_time.get('Groups', []):
                service_name = group.get('Keys', ['Unknown'])[0]
                metrics = group.get('Metrics', {})
                
                cost = float(metrics.get('BlendedCost', {}).get('Amount', 0))
                usage = float(metrics.get('UsageQuantity', {}).get('Amount', 0))
                
                gpu_tpu_type = None
                if 'p2' in service_name or 'p3' in service_name or 'p4' in service_name:
                    gpu_tpu_type = service_name
                
                record = {
                    'model_name': model_name or 'unknown',
                    'version': version,
                    'cloud_provider': 'AWS',
                    'service_name': service_name,
                    'gpu_tpu_type': gpu_tpu_type,
                    'gpu_tpu_usage': usage,
                    'cost': cost,
                    'start_date': start_date,
                    'end_date': end_date
                }
                
                results.append(record)
        
        return results
    
    def save_costs_to_db(self, cost_records):
        """
        Save cost records to the database.
        
        Args:
            cost_records (list): List of cost records
        
        Returns:
            int: Number of records saved
        """
        records_saved = 0
        
        for record in cost_records:
            try:
                insert_model_cost(
                    model_name=record['model_name'],
                    version=record['version'],
                    cloud_provider=record['cloud_provider'],
                    gpu_tpu_type=record['gpu_tpu_type'],
                    gpu_tpu_usage=record['gpu_tpu_usage'],
                    cost=record['cost'],
                    start_date=record['start_date'],
                    end_date=record['end_date'],
                    additional_info=json.dumps({'service_name': record['service_name']})
                )
                records_saved += 1
            except Exception as e:
                print(f"Error saving record to database: {str(e)}")
        
        return records_saved
    
    def export_costs_to_csv(self, cost_records, output_path=None):
        """
        Export cost records to CSV.
        
        Args:
            cost_records (list): List of cost records
            output_path (str, optional): Output path for CSV file
        
        Returns:
            str: Path to the saved CSV file
        """
        df = pd.DataFrame(cost_records)
        output_path = output_path or f"aws_costs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        return output_path

def main():
    """Main function to run AWS cost tracking."""
    parser = argparse.ArgumentParser(description='Track AWS GPU/TPU usage costs')
    parser.add_argument('--profile', type=str, help='AWS profile name')
    parser.add_argument('--region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--granularity', type=str, default='DAILY', choices=['DAILY', 'MONTHLY'],
                        help='Time granularity')
    parser.add_argument('--model-name', type=str, help='Model name to filter by tags')
    parser.add_argument('--save-to-db', action='store_true', help='Save costs to database')
    parser.add_argument('--output', type=str, help='Output path for CSV file')
    parser.add_argument('--use-mock-data', action='store_true', help='Use mock data instead of real AWS API')
    
    args = parser.parse_args()
    
    try:
        # Initialize AWS Cost Tracker
        cost_tracker = AWSCostTracker(profile=args.profile, region=args.region)
        
        # Get EC2 GPU costs
        print("Fetching EC2 GPU costs...")
        ec2_response = cost_tracker.get_ec2_gpu_costs(
            start_date=args.start_date,
            end_date=args.end_date,
            granularity=args.granularity
        )
        
        # Get SageMaker costs
        print("Fetching SageMaker costs...")
        sagemaker_response = cost_tracker.get_sagemaker_costs(
            start_date=args.start_date,
            end_date=args.end_date,
            granularity=args.granularity,
            model_name=args.model_name
        )
        
        # Parse responses
        ec2_records = cost_tracker.parse_cost_response(ec2_response, model_name=args.model_name)
        sagemaker_records = cost_tracker.parse_cost_response(sagemaker_response, model_name=args.model_name)
        
        # Combine records
        all_records = ec2_records + sagemaker_records
        
        # Print total cost
        total_cost = sum(record['cost'] for record in all_records)
        print(f"Total AWS cost: ${total_cost:.2f}")
        
        # Save to database if requested
        if args.save_to_db:
            records_saved = cost_tracker.save_costs_to_db(all_records)
            print(f"Saved {records_saved} records to database")
        
        # Export to CSV if requested
        if args.output:
            csv_path = cost_tracker.export_costs_to_csv(all_records, args.output)
            print(f"Exported costs to {csv_path}")
    
    except Exception as e:
        if "Unable to locate credentials" in str(e) or args.use_mock_data:
            print("AWS credentials not found or mock data requested. Using mock data.")
            
            if generate_mock_aws_costs:
                # Generate mock data
                mock_response = generate_mock_aws_costs(
                    start_date=args.start_date,
                    end_date=args.end_date,
                    model_name=args.model_name
                )
                
                # Initialize AWS Cost Tracker for parsing
                cost_tracker = AWSCostTracker(profile=args.profile, region=args.region)
                
                # Parse mock responses
                all_records = cost_tracker.parse_cost_response(mock_response, model_name=args.model_name)
                
                # Print total cost
                total_cost = sum(record['cost'] for record in all_records)
                print(f"Total AWS cost (mock data): ${total_cost:.2f}")
                
                # Save to database if requested
                if args.save_to_db:
                    records_saved = cost_tracker.save_costs_to_db(all_records)
                    print(f"Saved {records_saved} mock records to database")
                
                # Export to CSV if requested
                if args.output:
                    csv_path = cost_tracker.export_costs_to_csv(all_records, args.output)
                    print(f"Exported mock costs to {csv_path}")
            else:
                print("Error: Mock data module not available. Please run 'python main.py demo' instead.")
        else:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 