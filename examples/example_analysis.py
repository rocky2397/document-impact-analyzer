#!/usr/bin/env python
"""
Example script demonstrating how to use Document Impact Analyzer

This example shows how to analyze corporate documents for protectionism content,
calculating impact scores and generating reports and visualizations.
"""

import os
import sys
from document_impact_analyzer import DataExtractor, TextAnalyzer, ImpactAnalyzer

def main():
    """Run the document impact analysis example"""
    
    # Set up directories
    data_dir = os.path.join(os.path.dirname(__file__), 'examples', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), 'examples', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Document Impact Analyzer - Example Analysis")
    print("===========================================")
    
    # Step 1: Extract documents
    print("\nStep 1: Extracting documents from PDF files...")
    extractor = DataExtractor(data_dir=data_dir)
    documents = extractor.extract_documents(file_pattern="*.pdf")
    print(f"Extracted {len(documents)} document chunks")
    
    # Step 2: Analyze documents for the chosen topic (protectionism by default)
    print("\nStep 2: Analyzing documents for relevant content...")
    analyzer = TextAnalyzer()  # Default topic is protectionism
    
    # You can also specify a different topic:
    # analyzer = TextAnalyzer(topic="climate_change")
    
    # Or use custom keywords:
    # custom_keywords = ["supply chain", "logistics", "inventory", "warehouse", "delivery"]
    # analyzer = TextAnalyzer(custom_keywords=custom_keywords)
    
    # Find relevant content
    relevant_docs = analyzer.identify_relevant_content(documents)
    
    # Analyze sentiment
    analyzed_docs = analyzer.analyze_sentiment(relevant_docs)
    
    # Export labeled data
    labeled_data_path = os.path.join(output_dir, "labeled_data.csv")
    labeled_data = analyzer.export_labeled_data(analyzed_docs, output_path=labeled_data_path)
    print(f"Labeled data exported to {labeled_data_path}")
    
    # Step 3: Calculate impact scores and generate visualizations
    print("\nStep 3: Generating impact analysis and visualizations...")
    impact = ImpactAnalyzer(output_dir=output_dir)
    impact.engineer_features(labeled_data)
    
    # Generate visualizations
    impact.generate_visualizations()
    print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")
    
    # Step 4: Generate company reports
    print("\nStep 4: Generating company reports...")
    if impact.company_scores is not None:
        for company in impact.company_scores['company'].unique():
            report = impact.generate_company_report(company)
            print(f"Report generated for {company}")
    
    print("\nAnalysis complete! Results saved to:", output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())