#!/usr/bin/env python
"""
Example script showing how to analyze documents for climate change impacts
"""

import os
import sys
from document_impact_analyzer import DataExtractor, TextAnalyzer, ImpactAnalyzer

def main():
    """Run a climate change impact analysis"""
    
    # Set up directories
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'climate_change')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Document Impact Analyzer - Climate Change Analysis")
    print("=================================================")
    
    # Load data from sample CSV file
    # In a real scenario, you'd extract from PDFs using:
    # extractor = DataExtractor(data_dir=data_dir)
    # documents = extractor.extract_documents(file_pattern="*.pdf")
    
    sample_data_path = os.path.join(data_dir, "sample_labeled_data.csv")
    
    # Step 1: Analyze documents for climate change content
    print("\nStep 1: Analyzing documents for climate change content...")
    analyzer = TextAnalyzer(topic="climate_change")
    
    # If you wanted to use custom keywords instead of predefined ones:
    # custom_keywords = [
    #     "climate change", "global warming", "carbon emissions", "greenhouse gas",
    #     "renewable energy", "sustainability", "net zero", "carbon footprint"
    # ]
    # analyzer = TextAnalyzer(custom_keywords=custom_keywords)
    
    # Since we're using pre-labeled data, we'll skip the content identification step
    # and load directly from the sample data
    
    # Step 2: Generate impact analysis for climate change
    print("\nStep 2: Generating climate change impact analysis...")
    impact = ImpactAnalyzer(topic="climate_change", output_dir=output_dir)
    
    # Engineer features and calculate scores
    impact.engineer_features(sample_data_path)
    
    # Generate visualizations
    impact.generate_visualizations()
    print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")
    
    # Generate company reports
    print("\nStep 3: Generating company climate change reports...")
    if impact.company_scores is not None:
        for company in impact.company_scores['company'].unique():
            report = impact.generate_company_report(company)
            print(f"Climate change report generated for {company}")
    
    print("\nAnalysis complete! Results saved to:", output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())