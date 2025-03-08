#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the AI Cost Optimizer application.

This script provides a unified way to run different components of the application.
"""

import os
import sys
import argparse
from subprocess import run
import webbrowser
from threading import Timer

def open_browser(port=8501):
    """Open the browser with the Streamlit app."""
    webbrowser.open_new(f"http://localhost:{port}")

def run_dashboard():
    """Run the Streamlit dashboard."""
    dashboard_path = os.path.join('dashboard', 'app.py')
    # Open browser after a delay to ensure Streamlit is running
    Timer(2, open_browser).start()
    run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=True)

def init_database():
    """Initialize the database with sample data."""
    database_path = os.path.join('cost_tracking', 'database.py')
    run([sys.executable, database_path], check=True)

def run_model_comparison(models=None, save_to_db=True):
    """Run model comparison."""
    comparison_path = os.path.join('models', 'model_comparison.py')
    cmd = [sys.executable, comparison_path]
    
    if models:
        cmd.extend(["--models", models])
    
    if save_to_db:
        cmd.append("--save-to-db")
    
    run(cmd, check=True)

def run_onnx_optimization(model_name, optimize=True, save_to_db=True):
    """Run ONNX optimization."""
    optimization_path = os.path.join('models', 'onnx_optimization.py')
    cmd = [sys.executable, optimization_path, "--model-name", model_name]
    
    if optimize:
        cmd.append("--optimize")
    
    if save_to_db:
        cmd.append("--save-to-db")
    
    run(cmd, check=True)

def main():
    """Parse command line arguments and run the appropriate component."""
    parser = argparse.ArgumentParser(description='AI Cost Optimizer')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run the Streamlit dashboard')
    
    # Initialize database command
    init_parser = subparsers.add_parser('init', help='Initialize the database with sample data')
    
    # Model comparison command
    compare_parser = subparsers.add_parser('compare', help='Run model comparison')
    compare_parser.add_argument('--models', type=str, default='bert-base-uncased,distilbert-base-uncased,prajjwal1/bert-tiny',
                           help='Comma-separated list of models to compare')
    compare_parser.add_argument('--no-save', action='store_true', help='Do not save results to database')
    
    # ONNX optimization command
    optimize_parser = subparsers.add_parser('optimize', help='Run ONNX optimization')
    optimize_parser.add_argument('--model-name', type=str, required=True, help='Model name to optimize')
    optimize_parser.add_argument('--no-optimize', action='store_true', help='Do not create optimized ONNX model')
    optimize_parser.add_argument('--no-save', action='store_true', help='Do not save results to database')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate command
    if args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'init':
        init_database()
    elif args.command == 'compare':
        run_model_comparison(args.models, not args.no_save)
    elif args.command == 'optimize':
        run_onnx_optimization(args.model_name, not args.no_optimize, not args.no_save)
    else:
        # Default to showing help
        parser.print_help()

if __name__ == "__main__":
    main() 