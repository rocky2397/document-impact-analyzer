#!/usr/bin/env python
"""
Utility script to compress and convert annual report PDFs from the original data directory.

This script extracts text from the PDFs in the original data directory and 
saves it in compressed formats suitable for distribution with the package.
"""

import os
import sys
import gzip
import pandas as pd
import shutil
from document_impact_analyzer.data_extraction import DataExtractor

def compress_pdfs_for_github(original_data_dir, output_dir, companies=None):
    """
    Extract text from PDFs and save as compressed files for GitHub distribution
    
    Args:
        original_data_dir (str): Directory containing original PDF files
        output_dir (str): Directory to save compressed files
        companies (list, optional): List of companies to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Default companies if none specified
    if companies is None:
        companies = ["BP", "CHEVRON", "EON", "RWE", "NEXTERA"]
    
    print(f"Processing PDF files from {original_data_dir}...")
    
    # Use the built-in extractor to get text
    extractor = DataExtractor(data_dir=original_data_dir)
    
    # Process each company
    for company in companies:
        print(f"\nProcessing {company} documents...")
        
        try:
            # Extract documents for this company
            documents = extractor.extract_documents(
                companies=[company],
                file_pattern="*.pdf"
            )
            
            if not documents:
                print(f"No documents found for {company}")
                continue
                
            print(f"Extracted {len(documents)} document chunks")
            
            # Group by year
            year_docs = {}
            for doc in documents:
                year = doc.metadata.get("year", "UNKNOWN")
                
                if year not in year_docs:
                    year_docs[year] = []
                
                year_docs[year].append({
                    "content": doc.page_content,
                    "company": company,
                    "year": year,
                    "file_name": doc.metadata.get("file_name", ""),
                    "type": "text"
                })
            
            # Save each year as a separate compressed file
            for year, docs in year_docs.items():
                # Create DataFrame
                df = pd.DataFrame(docs)
                
                # Save as compressed CSV
                csv_output_path = os.path.join(output_dir, f"{company}_{year}.csv.gz")
                with gzip.open(csv_output_path, 'wt') as f:
                    df.to_csv(f, index=False)
                
                print(f"Saved {len(docs)} chunks to {csv_output_path}")
                
                # Also save the original file size for reference
                original_file = f"{company}/{company}_{year}.pdf"
                if os.path.exists(os.path.join(original_data_dir, original_file)):
                    original_size = os.path.getsize(os.path.join(original_data_dir, original_file))
                    compressed_size = os.path.getsize(csv_output_path)
                    
                    print(f"  Original PDF: {original_size/1024/1024:.2f} MB")
                    print(f"  Compressed CSV: {compressed_size/1024/1024:.2f} MB")
                    print(f"  Compression ratio: {original_size/compressed_size:.1f}x")
                
        except Exception as e:
            print(f"Error processing {company}: {e}")
    
    # Create a merged sample file with a few examples from each company
    print("\nCreating sample data file with excerpts from all companies...")
    create_sample_file(output_dir, companies)
    
    print("\nCompression complete!")
    return output_dir

def create_sample_file(compressed_dir, companies, sample_output="sample_labeled_data.csv"):
    """Create a sample file with a few examples from each company for testing"""
    all_samples = []
    
    # For each compressed file, read a few samples
    for filename in os.listdir(compressed_dir):
        if not filename.endswith(".csv.gz"):
            continue
            
        try:
            file_path = os.path.join(compressed_dir, filename)
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
                
            # Take a few samples from each file (max 5)
            samples = df.sample(min(5, len(df)))
            
            # For the sample, we'll add placeholder sentiment scores
            if "sentiment_label" not in samples.columns:
                samples["sentiment_label"] = "NEGATIVE"
                samples["sentiment_score"] = 0.8
            
            # For the sample, we'll add placeholder relevance scores
            if "relevance_score" not in samples.columns:
                samples["relevance_score"] = samples.apply(
                    lambda x: 0.1 + 0.3 * hash(x["content"]) % 100 / 100, 
                    axis=1
                )
                
            all_samples.append(samples)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Combine all samples
    if all_samples:
        combined_df = pd.concat(all_samples, ignore_index=True)
        
        # Save the sample file
        sample_path = os.path.join(compressed_dir, sample_output)
        combined_df.to_csv(sample_path, index=False)
        print(f"Created sample file with {len(combined_df)} entries: {sample_path}")
        
        # Copy to examples/data directory
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
        os.makedirs(examples_dir, exist_ok=True)
        shutil.copy(sample_path, os.path.join(examples_dir, sample_output))
        print(f"Copied sample file to examples/data directory")
    else:
        print("No samples could be created")

def main():
    """Run the compression script"""
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Compress and convert PDF annual reports for GitHub distribution")
    parser.add_argument("--input-dir", default="/Users/rockyauer/Downloads/RavenPack_assignment/data", 
                        help="Directory containing original PDF files")
    parser.add_argument("--output-dir", default="./compressed_data", 
                        help="Directory to save compressed files")
    parser.add_argument("--companies", nargs="+", 
                        help="List of companies to process (default: BP CHEVRON EON RWE NEXTERA)")
    
    args = parser.parse_args()
    
    # Run the compression
    compress_pdfs_for_github(args.input_dir, args.output_dir, args.companies)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())