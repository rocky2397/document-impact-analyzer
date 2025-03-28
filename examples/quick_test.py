#!/usr/bin/env python
"""
Quick test script to verify that the document impact analyzer package is working correctly.
"""

import os
import sys
from document_impact_analyzer import TextAnalyzer, ImpactAnalyzer

def main():
    """Run a quick test to verify package functionality"""
    
    print("Document Impact Analyzer - Quick Test")
    print("=====================================")
    
    # Set up directories
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to sample data
    sample_data_path = os.path.join(data_dir, "sample_labeled_data.csv")
    
    if not os.path.exists(sample_data_path):
        print(f"ERROR: Sample data file not found at {sample_data_path}")
        print("Please make sure you have the required sample data file in the examples/data directory.")
        return 1
    
    print(f"\nFound sample data: {sample_data_path}")
    
    try:
        # Step 1: Test the TextAnalyzer functionality
        print("\nTesting TextAnalyzer component...")
        analyzer = TextAnalyzer(topic="protectionism")
        
        # Check if keywords are loaded correctly
        print(f"Loaded {len(analyzer.keywords)} keywords for protectionism topic")
        print("Sample keywords:", analyzer.keywords[:5])
        
        # Step 2: Test the ImpactAnalyzer functionality with the sample data
        print("\nTesting ImpactAnalyzer component...")
        impact = ImpactAnalyzer(topic="protectionism", output_dir=output_dir)
        
        # Process the sample data
        feature_df = impact.engineer_features(sample_data_path)
        
        if feature_df is not None:
            print(f"Successfully processed sample data with {len(feature_df)} entries")
            
            # Generate a test visualization
            print("\nGenerating test visualization...")
            impact.generate_visualizations()
            
            # Generate a sample report
            if impact.company_scores is not None:
                company = impact.company_scores['company'].iloc[0]
                print(f"\nGenerating test report for {company}...")
                impact.generate_company_report(company)
                
        print("\n✅ Test completed successfully!")
        print(f"Output files have been saved to: {output_dir}")
        
        return 0
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())