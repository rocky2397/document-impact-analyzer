#!/usr/bin/env python
"""Simple test script to verify the document_impact_analyzer package is working"""

import os
from document_impact_analyzer import TextAnalyzer, ImpactAnalyzer

# Print confirmation that the package is loaded
print("Successfully imported document_impact_analyzer package")

# Test the TextAnalyzer component
print("\nTesting TextAnalyzer functionality...")
analyzer = TextAnalyzer(topic="protectionism")
print(f"Loaded {len(analyzer.keywords)} keywords for protectionism analysis")
print("Sample keywords:", analyzer.keywords[:5])

# Print the current directory to help with path resolution
print("\nCurrent directory:", os.getcwd())

# Check if the sample data exists
sample_data_path = os.path.join("examples", "data", "sample_labeled_data.csv")
if os.path.exists(sample_data_path):
    print(f"Found sample data at: {sample_data_path}")
else:
    print(f"Sample data not found at: {sample_data_path}")
    # List files in the examples/data directory
    data_dir = os.path.join("examples", "data")
    if os.path.exists(data_dir):
        print(f"Files in {data_dir}:")
        for file in os.listdir(data_dir):
            print(f"  - {file}")
    else:
        print(f"Directory not found: {data_dir}")

print("\nTest completed")