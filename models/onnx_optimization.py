#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ONNX optimization module for AI Cost Optimizer.

This module provides functionality to:
1. Convert PyTorch/TensorFlow models to ONNX format
2. Optimize ONNX models for inference
3. Compare performance before and after optimization
"""

import os
import argparse
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort
import onnx

# Add path to parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_tracking.database import insert_onnx_optimization_metrics

def convert_to_onnx(model_name, output_path, task="sequence-classification", num_labels=2):
    """
    Convert a transformers model to ONNX format.
    
    Args:
        model_name (str): The name or path of the transformer model to convert
        output_path (str): The path to save the ONNX model
        task (str): The task the model will be used for
        num_labels (int): Number of labels for sequence classification
    
    Returns:
        str: Path to the saved ONNX model
    """
    print(f"Converting {model_name} to ONNX format...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load appropriate model based on task
    if task == "sequence-classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        model = AutoModel.from_pretrained(model_name)
    
    # Create dummy input for the model
    dummy_input_text = "This is a sample input for ONNX conversion."
    dummy_inputs = tokenizer(dummy_input_text, return_tensors="pt")
    
    # Get input names and dynamic axes
    input_names = list(dummy_inputs.keys())
    dynamic_axes = {k: {0: "batch_size"} for k in input_names}
    dynamic_axes["output"] = {0: "batch_size"}
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            output_path,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=12,
            do_constant_folding=True,
            verbose=False
        )
    
    print(f"Model converted and saved to {output_path}")
    return output_path

def optimize_onnx_model(input_path, output_path=None):
    """
    Optimize an ONNX model for inference.
    
    Args:
        input_path (str): The path to the ONNX model to optimize
        output_path (str): The path to save the optimized ONNX model
    
    Returns:
        str: Path to the optimized ONNX model
    """
    if output_path is None:
        output_path = input_path.replace(".onnx", "_optimized.onnx")
    
    print(f"Optimizing ONNX model {input_path}...")
    
    # Load the model
    model = onnx.load(input_path)
    
    # Check the model
    onnx.checker.check_model(model)
    
    # Optimize the model
    from onnxruntime.transformers import optimizer
    optimized_model = optimizer.optimize_model(
        input_path,
        model_type='bert',
        num_heads=12,
        hidden_size=768
    )
    
    # Save the optimized model
    optimized_model.save_model_to_file(output_path)
    
    print(f"Optimized model saved to {output_path}")
    return output_path

def create_onnx_session(model_path, execution_provider="CPUExecutionProvider"):
    """
    Create an ONNX Runtime session for a model.
    
    Args:
        model_path (str): Path to the ONNX model
        execution_provider (str): ONNX Runtime execution provider
    
    Returns:
        ort.InferenceSession: ONNX Runtime session
    """
    providers = [execution_provider]
    
    # Create session options
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create session
    session = ort.InferenceSession(model_path, session_options, providers=providers)
    
    return session

def compare_inference_performance(model_name, onnx_model_path, optimized_onnx_model_path=None, num_runs=10):
    """
    Compare inference performance between PyTorch, ONNX, and optimized ONNX models.
    
    Args:
        model_name (str): The name or path of the transformer model
        onnx_model_path (str): Path to the ONNX model
        optimized_onnx_model_path (str): Path to the optimized ONNX model
        num_runs (int): Number of inference runs to average over
    
    Returns:
        dict: Dictionary of performance metrics
    """
    print(f"Comparing inference performance for {model_name}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Create ONNX session
    onnx_session = create_onnx_session(onnx_model_path)
    
    # Create optimized ONNX session if available
    if optimized_onnx_model_path:
        optimized_onnx_session = create_onnx_session(optimized_onnx_model_path)
    
    # Sample inputs
    sample_texts = [
        "This is a sample text for inference performance comparison.",
        "ONNX Runtime provides excellent performance for model inference.",
        "Model optimization can significantly reduce inference time.",
        "AI cost optimization is becoming increasingly important.",
        "PyTorch models can be converted to ONNX format for better performance."
    ]
    
    # Tokenize inputs
    inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")
    input_names = list(inputs.keys())
    onnx_inputs = {k: v.numpy() for k, v in inputs.items()}
    
    # Run PyTorch inference
    pytorch_times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        pytorch_times.append(time.time() - start_time)
    
    # Run ONNX inference
    onnx_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = onnx_session.run(None, onnx_inputs)
        onnx_times.append(time.time() - start_time)
    
    # Run optimized ONNX inference if available
    optimized_onnx_times = []
    if optimized_onnx_model_path:
        for _ in range(num_runs):
            start_time = time.time()
            _ = optimized_onnx_session.run(None, onnx_inputs)
            optimized_onnx_times.append(time.time() - start_time)
    
    # Calculate averages
    avg_pytorch_time = sum(pytorch_times) / len(pytorch_times)
    avg_onnx_time = sum(onnx_times) / len(onnx_times)
    avg_optimized_onnx_time = sum(optimized_onnx_times) / len(optimized_onnx_times) if optimized_onnx_times else None
    
    # Calculate speedup
    onnx_speedup = avg_pytorch_time / avg_onnx_time
    optimized_onnx_speedup = avg_pytorch_time / avg_optimized_onnx_time if avg_optimized_onnx_time else None
    
    # Collect metrics
    metrics = {
        "model_name": model_name,
        "pytorch_inference_time": avg_pytorch_time,
        "onnx_inference_time": avg_onnx_time,
        "optimized_onnx_inference_time": avg_optimized_onnx_time,
        "onnx_speedup": onnx_speedup,
        "optimized_onnx_speedup": optimized_onnx_speedup
    }
    
    print(f"PyTorch inference time: {avg_pytorch_time:.4f} seconds")
    print(f"ONNX inference time: {avg_onnx_time:.4f} seconds (speedup: {onnx_speedup:.2f}x)")
    if avg_optimized_onnx_time:
        print(f"Optimized ONNX inference time: {avg_optimized_onnx_time:.4f} seconds (speedup: {optimized_onnx_speedup:.2f}x)")
    
    return metrics

def plot_performance_comparison(metrics):
    """
    Plot performance comparison between PyTorch, ONNX, and optimized ONNX models.
    
    Args:
        metrics (dict): Dictionary of performance metrics
    """
    # Create bar chart
    labels = ['PyTorch', 'ONNX', 'Optimized ONNX']
    times = [
        metrics['pytorch_inference_time'], 
        metrics['onnx_inference_time'],
        metrics['optimized_onnx_inference_time'] if metrics['optimized_onnx_inference_time'] is not None else 0
    ]
    
    # Remove optimized ONNX if not available
    if metrics['optimized_onnx_inference_time'] is None:
        labels = labels[:-1]
        times = times[:-1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times)
    
    # Add speedup labels
    for i, bar in enumerate(bars):
        if i > 0:  # Skip PyTorch (base)
            speedup = metrics['onnx_speedup'] if i == 1 else metrics['optimized_onnx_speedup']
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{speedup:.2f}x',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
    
    plt.title(f'Inference Time Comparison for {metrics["model_name"]}')
    plt.ylabel('Inference Time (seconds)')
    plt.ylim(0, max(times) * 1.2)  # Add some headroom for the text
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('onnx_performance_comparison.png')
    plt.show()

def main():
    """Main function to run ONNX optimization."""
    parser = argparse.ArgumentParser(description='Convert and optimize transformer models to ONNX')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased',
                        help='Name or path of the transformer model')
    parser.add_argument('--task', type=str, default='sequence-classification',
                        help='Task the model will be used for')
    parser.add_argument('--output', type=str, default='../data/models',
                        help='Output directory for ONNX models')
    parser.add_argument('--optimize', action='store_true',
                        help='Whether to also create an optimized version of the ONNX model')
    parser.add_argument('--save-to-db', action='store_true',
                        help='Whether to save the optimization metrics to the SQLite database')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Define output paths
    onnx_model_path = os.path.join(args.output, f"{args.model_name.split('/')[-1]}.onnx")
    optimized_onnx_model_path = os.path.join(args.output, f"{args.model_name.split('/')[-1]}_optimized.onnx")
    
    # Convert model to ONNX
    convert_to_onnx(args.model_name, onnx_model_path, task=args.task)
    
    # Optimize ONNX model if requested
    if args.optimize:
        optimize_onnx_model(onnx_model_path, optimized_onnx_model_path)
    else:
        optimized_onnx_model_path = None
    
    # Compare performance
    metrics = compare_inference_performance(
        args.model_name,
        onnx_model_path,
        optimized_onnx_model_path
    )
    
    # Plot performance comparison
    plot_performance_comparison(metrics)
    
    # Save optimization metrics to database if requested
    if args.save_to_db:
        try:
            from cost_tracking.database import insert_onnx_optimization_metrics
            insert_onnx_optimization_metrics(
                model_name=metrics['model_name'],
                pytorch_time=metrics['pytorch_inference_time'],
                onnx_time=metrics['onnx_inference_time'],
                optimized_onnx_time=metrics['optimized_onnx_inference_time'],
                onnx_speedup=metrics['onnx_speedup'],
                optimized_onnx_speedup=metrics['optimized_onnx_speedup']
            )
            print("Optimization metrics saved to database")
        except Exception as e:
            print(f"Error saving to database: {str(e)}")

if __name__ == "__main__":
    main() 