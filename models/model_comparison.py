#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model comparison module for AI Cost Optimizer.

This module compares different transformer models in terms of:
- Inference speed
- Memory usage
- Parameter count
- Accuracy on a sample dataset
"""

import argparse
import time
import os
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

# Add path to parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cost_tracking.database import insert_model_metrics

def get_model_size(model):
    """Calculate the model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model_performance(model_name, task="sequence-classification", dataset=None):
    """
    Evaluate model performance in terms of speed, memory usage, and accuracy.
    
    Args:
        model_name (str): The name or path of the transformer model to evaluate
        task (str): The task to evaluate on (e.g., "sequence-classification")
        dataset (list): Optional list of (text, label) tuples for evaluation
    
    Returns:
        dict: A dictionary containing performance metrics
    """
    # Use a sample dataset if none provided
    if dataset is None:
        dataset = [
            ("This movie is great! I really enjoyed it.", 1),
            ("The acting was terrible and the plot made no sense.", 0),
            ("The cinematography was beautiful but the story was lacking.", 0),
            ("I would highly recommend this film to anyone.", 1),
            ("This was an absolute waste of time.", 0)
        ]
    
    print(f"Evaluating model: {model_name}")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # For sequence classification
        if task == "sequence-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        # Get model size
        model_size_mb = get_model_size(model)
        param_count = count_parameters(model)
        
        # Measure inference time
        texts = [item[0] for item in dataset]
        labels = [item[1] for item in dataset]
        
        # Tokenize inputs
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Measure GPU memory before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            model = model.cuda()
            inputs = {k: v.cuda() for k, v in inputs.items()}
            initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        inference_time = time.time() - start_time
        
        # Calculate memory usage
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            memory_usage = peak_memory - initial_memory
        else:
            memory_usage = None
        
        # Calculate accuracy for classification models
        if task == "sequence-classification":
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            accuracy = accuracy_score(labels, predictions)
        else:
            accuracy = None
        
        # Collect metrics
        metrics = {
            "model_name": model_name,
            "inference_time_seconds": inference_time,
            "memory_usage_mb": memory_usage,
            "model_size_mb": model_size_mb,
            "parameter_count": param_count,
            "accuracy": accuracy
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error evaluating model {model_name}: {str(e)}")
        return {
            "model_name": model_name,
            "error": str(e)
        }

def save_comparison_results(results, output_path="model_comparison_results.json"):
    """Save comparison results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")

def plot_comparison(results):
    """Plot comparison of model performance metrics."""
    df = pd.DataFrame(results)
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot inference time
    axs[0, 0].bar(df['model_name'], df['inference_time_seconds'])
    axs[0, 0].set_title('Inference Time (seconds)')
    axs[0, 0].set_ylabel('Time (s)')
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot model size
    axs[0, 1].bar(df['model_name'], df['model_size_mb'])
    axs[0, 1].set_title('Model Size (MB)')
    axs[0, 1].set_ylabel('Size (MB)')
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot parameter count
    axs[1, 0].bar(df['model_name'], df['parameter_count'] / 1_000_000)
    axs[1, 0].set_title('Parameter Count (millions)')
    axs[1, 0].set_ylabel('Parameters (M)')
    axs[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot accuracy if available
    if 'accuracy' in df.columns and not df['accuracy'].isnull().all():
        axs[1, 1].bar(df['model_name'], df['accuracy'])
        axs[1, 1].set_title('Accuracy')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].tick_params(axis='x', rotation=45)
        axs[1, 1].set_ylim(0, 1)
    else:
        axs[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def main():
    """Main function to run model comparison."""
    parser = argparse.ArgumentParser(description='Compare different transformer models')
    parser.add_argument('--models', type=str, default='bert-base-uncased,distilbert-base-uncased,prajjwal1/bert-tiny',
                        help='Comma-separated list of model names to compare')
    parser.add_argument('--task', type=str, default='sequence-classification',
                        help='Task to evaluate (sequence-classification, embedding, etc.)')
    parser.add_argument('--output', type=str, default='../data/model_comparison_results.json',
                        help='Output path for the comparison results')
    parser.add_argument('--save-to-db', action='store_true',
                        help='Whether to save the results to the SQLite database')
    
    args = parser.parse_args()
    
    # Parse model names
    model_names = [name.strip() for name in args.models.split(',')]
    
    # Run comparisons
    results = []
    for model_name in model_names:
        metrics = evaluate_model_performance(model_name, task=args.task)
        results.append(metrics)
        
        # Insert into database if requested
        if args.save_to_db and 'error' not in metrics:
            try:
                print(f"Saving metrics for {model_name} to database")
                insert_model_metrics(
                    model_name=metrics['model_name'],
                    inference_time=metrics['inference_time_seconds'],
                    model_size=metrics['model_size_mb'],
                    parameter_count=metrics['parameter_count'],
                    accuracy=metrics['accuracy']
                )
            except Exception as e:
                print(f"Error saving to database: {str(e)}")
    
    # Save results
    save_comparison_results(results, args.output)
    
    # Plot comparison
    plot_comparison(results)

if __name__ == "__main__":
    main() 