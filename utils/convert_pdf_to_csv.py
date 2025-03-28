#!/usr/bin/env python
"""
Utility script to convert PDF annual reports to gzipped CSV files.
This makes it easier to share example data without distributing large PDF files.
"""

import os
import gzip
import pandas as pd
import argparse
from document_impact_analyzer.data_extraction import DataExtractor

def convert_pdfs_to_csv_gz(input_dir, output_dir, companies=None, years=None):
    """
    Extract text from PDFs and save as gzipped CSV files
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save CSV files
        companies (list, optional): List of companies to process
        years (list, optional): List of years to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting text from PDFs in {input_dir}...")
    extractor = DataExtractor(data_dir=input_dir)
    documents = extractor.extract_documents(
        companies=companies,
        years=years,
        file_pattern="*.pdf"
    )
    
    if not documents:
        print("No documents found matching the criteria.")
        return
    
    print(f"Extracted {len(documents)} document chunks")
    
    # Group documents by company and year
    grouped_docs = {}
    for doc in documents:
        company = doc.metadata.get("source", "UNKNOWN")
        year = doc.metadata.get("year", "UNKNOWN")
        key = f"{company}_{year}"
        
        if key not in grouped_docs:
            grouped_docs[key] = []
        
        grouped_docs[key].append({
            "content": doc.page_content,
            "company": company,
            "year": year,
            "file_name": doc.metadata.get("file_name", ""),
            "chunk_id": doc.metadata.get("chunk_id", 0)
        })
    
    # Save each company-year combination as a separate CSV file
    for key, docs in grouped_docs.items():
        output_file = os.path.join(output_dir, f"{key}.csv.gz")
        df = pd.DataFrame(docs)
        
        # Save as gzipped CSV
        with gzip.open(output_file, 'wt') as f:
            df.to_csv(f, index=False)
        
        print(f"Saved {len(docs)} chunks to {output_file}")

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Convert PDF annual reports to gzipped CSV files")
    parser.add_argument("--input-dir", default="../data", help="Directory containing PDF files")
    parser.add_argument("--output-dir", default="./examples/data", help="Directory to save CSV files")
    parser.add_argument("--companies", nargs="+", help="List of companies to process")
    parser.add_argument("--years", nargs="+", help="List of years to process")
    
    args = parser.parse_args()
    
    convert_pdfs_to_csv_gz(
        args.input_dir, 
        args.output_dir,
        args.companies,
        args.years
    )
    
    return 0

if __name__ == "__main__":
    main()