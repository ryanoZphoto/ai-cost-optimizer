#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Cloud Platform Cost Tracker module for AI Cost Optimizer.

This module provides functionality to:
1. Fetch GPU/TPU usage costs from GCP
2. Filter costs by date, service, and labels
3. Store cost data in the SQLite database
"""

import os
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import billing_v1
from google.cloud.billing import CloudCatalog, CloudBilling

# Add path to parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_tracking.database import insert_model_cost

class GCPCostTracker:
    """Class to track GCP GPU/TPU usage costs."""
    
    def __init__(self, credentials_path=None):
        """
        Initialize GCP Cost Tracker.
        
        Args:
            credentials_path (str, optional): Path to GCP credentials JSON file
        """
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        self.billing_client = billing_v1.CloudBillingClient()
        self.catalog_client = billing_v1.CloudCatalogClient()
        self.current_date = datetime.now()
    
    def get_billing_accounts(self):
        """
        Get billing accounts for the authenticated user.
        
        Returns:
            list: List of billing accounts
        """
        billing_accounts = []
        
        request = billing_v1.ListBillingAccountsRequest()
        page_result = self.billing_client.list_billing_accounts(request=request)
        
        for response in page_result:
            billing_accounts.append({
                'name': response.name,
                'display_name': response.display_name,
                'open': response.open
            })
        
        return billing_accounts
    
    def get_costs(self, billing_account, start_date=None, end_date=None, 
                 filter_string=None, model_name=None):
        """
        Get GCP costs for GPU/TPU instances.
        
        Args:
            billing_account (str): Billing account ID
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            filter_string (str, optional): Filter string
            model_name (str, optional): Model name to filter by labels
        
        Returns:
            list: List of cost data entries
        """
        # Set default dates if not provided
        if not start_date:
            start_date = (self.current_date - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = self.current_date.strftime('%Y-%m-%d')
        
        # Ensure billing account is properly formatted
        if not billing_account.startswith('billingAccounts/'):
            billing_account = f'billingAccounts/{billing_account}'
        
        # Construct filter for GPU/TPU costs
        if not filter_string:
            service_filters = [
                'service.description:"Compute Engine"',
                'service.description:"Cloud TPU"',
                'service.description:"AI Platform"'
            ]
            
            # Add GPU/TPU specific filters
            resource_filters = [
                '(resource.type:"compute.googleapis.com/Instance" AND resource.labels.machine_type:(n1-standard-4-t4 OR a2-highgpu* OR n1-standard-8-t4))',
                'resource.type:"aiplatform.googleapis.com/Endpoint"',
                'resource.type:"tpu.googleapis.com/Node"'
            ]
            
            # Add model name filter if provided
            if model_name:
                model_filter = f'labels.key:"model_name" AND labels.value:"{model_name}"'
                filter_string = f'({" OR ".join(service_filters)}) AND ({" OR ".join(resource_filters)}) AND ({model_filter})'
            else:
                filter_string = f'({" OR ".join(service_filters)}) AND ({" OR ".join(resource_filters)})'
        
        request = billing_v1.QueryCostsRequest(
            billing_account=billing_account,
            date_period={
                'start_date': {
                    'year': int(start_date.split('-')[0]),
                    'month': int(start_date.split('-')[1]),
                    'day': int(start_date.split('-')[2])
                },
                'end_date': {
                    'year': int(end_date.split('-')[0]),
                    'month': int(end_date.split('-')[1]),
                    'day': int(end_date.split('-')[2])
                }
            },
            filter=filter_string,
            group_by=['project.id', 'resource.type', 'resource.labels.machine_type', 'service.description']
        )
        
        response = self.billing_client.query_costs(request=request)
        
        return self._parse_cost_response(response)
    
    def _parse_cost_response(self, response):
        """
        Parse GCP billing response.
        
        Args:
            response: GCP billing response
        
        Returns:
            list: List of cost records
        """
        results = []
        
        for cost_info in getattr(response, 'cost_breakdown', []):
            group_keys = {}
            
            for dimension in cost_info.dimension_value:
                group_keys[dimension.dimension] = dimension.value
            
            for c in cost_info.cost:
                # Extract cost in USD
                cost_usd = 0
                for r in c.aggregation:
                    if r.HasField('currency'):
                        if r.currency.currency_code == 'USD':
                            cost_usd = r.aggregation.value
            
            # Determine if it's a GPU or TPU
            machine_type = group_keys.get('resource.labels.machine_type', '')
            resource_type = group_keys.get('resource.type', '')
            service_desc = group_keys.get('service.description', '')
            
            gpu_tpu_type = None
            if 'tpu' in resource_type.lower():
                gpu_tpu_type = 'TPU'
            elif any(gpu_type in machine_type.lower() for gpu_type in ['gpu', 't4', 'p100', 'v100', 'a100']):
                gpu_tpu_type = machine_type
            
            record = {
                'model_name': 'unknown',  # Would need to extract from labels
                'version': None,
                'cloud_provider': 'GCP',
                'service_name': service_desc,
                'gpu_tpu_type': gpu_tpu_type,
                'gpu_tpu_usage': None,  # GCP doesn't provide usage hours directly
                'cost': cost_usd,
                'start_date': response.date_period.start_date.strftime('%Y-%m-%d'),
                'end_date': response.date_period.end_date.strftime('%Y-%m-%d'),
                'project_id': group_keys.get('project.id', ''),
                'resource_type': resource_type
            }
            
            results.append(record)
        
        return results
    
    def get_gpu_machine_types(self):
        """
        Get list of available GPU machine types in GCP.
        
        Returns:
            list: List of GPU machine types
        """
        gpu_machine_types = []
        
        request = billing_v1.ListServicesRequest()
        services = self.catalog_client.list_services(request=request)
        
        for service in services:
            if service.display_name == "Compute Engine":
                sku_request = billing_v1.ListSkusRequest(parent=service.name)
                skus = self.catalog_client.list_skus(request=sku_request)
                
                for sku in skus:
                    if any(gpu_term in sku.description.lower() for gpu_term in ['gpu', 'nvidia', 't4', 'p100', 'v100', 'a100']):
                        gpu_machine_types.append({
                            'name': sku.name,
                            'description': sku.description,
                            'service': service.display_name
                        })
        
        return gpu_machine_types
    
    def save_costs_to_db(self, cost_records, model_name=None, version=None):
        """
        Save cost records to the database.
        
        Args:
            cost_records (list): List of cost records
            model_name (str, optional): Model name
            version (str, optional): Model version
        
        Returns:
            int: Number of records saved
        """
        records_saved = 0
        
        for record in cost_records:
            try:
                record_model_name = model_name or record.get('model_name', 'unknown')
                record_version = version or record.get('version')
                
                insert_model_cost(
                    model_name=record_model_name,
                    version=record_version,
                    cloud_provider=record['cloud_provider'],
                    gpu_tpu_type=record['gpu_tpu_type'],
                    gpu_tpu_usage=record['gpu_tpu_usage'],
                    cost=record['cost'],
                    start_date=record['start_date'],
                    end_date=record['end_date'],
                    additional_info=json.dumps({
                        'service_name': record['service_name'],
                        'project_id': record.get('project_id', ''),
                        'resource_type': record.get('resource_type', '')
                    })
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
        output_path = output_path or f"gcp_costs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        return output_path

def main():
    """Main function to run GCP cost tracking."""
    parser = argparse.ArgumentParser(description='Track GCP GPU/TPU usage costs')
    parser.add_argument('--credentials', type=str, help='Path to GCP credentials JSON file')
    parser.add_argument('--billing-account', type=str, required=True, help='GCP billing account ID')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--filter', type=str, help='Custom filter string')
    parser.add_argument('--model-name', type=str, help='Model name to filter by labels')
    parser.add_argument('--save-to-db', action='store_true', help='Save costs to database')
    parser.add_argument('--output', type=str, help='Output path for CSV file')
    
    args = parser.parse_args()
    
    # Initialize GCP Cost Tracker
    cost_tracker = GCPCostTracker(credentials_path=args.credentials)
    
    # List billing accounts if no billing account provided
    if not args.billing_account:
        print("No billing account provided. Available billing accounts:")
        billing_accounts = cost_tracker.get_billing_accounts()
        for account in billing_accounts:
            print(f"- {account['name']} ({account['display_name']})")
        return
    
    # Get costs
    print(f"Fetching GCP costs for billing account {args.billing_account}...")
    cost_records = cost_tracker.get_costs(
        billing_account=args.billing_account,
        start_date=args.start_date,
        end_date=args.end_date,
        filter_string=args.filter,
        model_name=args.model_name
    )
    
    # Print total cost
    total_cost = sum(record['cost'] for record in cost_records)
    print(f"Total GCP cost: ${total_cost:.2f}")
    
    # Save to database if requested
    if args.save_to_db:
        records_saved = cost_tracker.save_costs_to_db(
            cost_records,
            model_name=args.model_name
        )
        print(f"Saved {records_saved} records to database")
    
    # Export to CSV if requested
    if args.output:
        csv_path = cost_tracker.export_costs_to_csv(cost_records, args.output)
        print(f"Exported costs to {csv_path}")

if __name__ == "__main__":
    main() 