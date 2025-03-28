#!/usr/bin/env python
"""
Example script showing how to fine-tune a model for topic impact analysis.

This script demonstrates the complete workflow:
1. Extract documents from PDF files
2. Identify relevant content related to the topic
3. Label the data with sentiment analysis
4. Fine-tune a model on the labeled data
5. Evaluate model performance and generate predictions
"""

import os
import sys
from document_impact_analyzer import (
    DataExtractor, 
    TextAnalyzer, 
    ImpactAnalyzer,
    ModelTrainer
)

def main():
    """Run the complete fine-tuning workflow"""
    
    # Set up directories
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'fine_tuned_model')
    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print("Document Impact Analyzer - Model Fine-Tuning Example")
    print("===================================================")
    
    # Step 1: Extract documents from PDF files
    print("\nStep 1: Extracting documents from files...")
    
    # Option 1: Extract from PDF files if you have them
    # extractor = DataExtractor(data_dir=data_dir)
    # documents = extractor.extract_documents(file_pattern="*.pdf")
    
    # Option 2: For this example, use the provided sample data
    sample_data_path = os.path.join(data_dir, "sample_labeled_data.csv")
    print(f"Using sample data from {sample_data_path}")
    
    # Step 2: Analyze documents for topic relevance
    print("\nStep 2: Analyzing documents for protectionism content...")
    analyzer = TextAnalyzer(topic="protectionism")  # Use the default protectionism topic
    
    # For real PDF data, you would do:
    # relevant_docs = analyzer.identify_relevant_content(documents)
    # analyzed_docs = analyzer.analyze_sentiment(relevant_docs)
    # labeled_data = analyzer.export_labeled_data(analyzed_docs, output_path=os.path.join(output_dir, "labeled_data.csv"))
    
    # For this example, we'll use the sample data directly since it's already labeled
    labeled_data_path = sample_data_path
    
    # Step 3: Fine-tune a model on the labeled data
    print("\nStep 3: Fine-tuning model on labeled data...")
    
    # Initialize model trainer
    # You can use different pretrained models:
    # - "distilbert-base-uncased" (smaller, faster)
    # - "bert-base-uncased" (more accurate)
    # - "roberta-base" (often better performance)
    trainer = ModelTrainer(
        topic_name="Protectionism", 
        model_name="distilbert-base-uncased",
        output_dir=model_dir
    )
    
    # Prepare dataset
    print("Preparing datasets...")
    train_dataset, eval_dataset = trainer.prepare_dataset(labeled_data_path)
    
    # Train the model
    print("Starting model training (this may take a while)...")
    trained_model = trainer.train(
        train_dataset, 
        eval_dataset, 
        epochs=3,  # Increase for better results
        batch_size=8   # Reduce if you encounter memory issues
    )
    
    # Step 4: Evaluate the model
    print("\nStep 4: Evaluating model performance...")
    eval_results = trainer.evaluate(trained_model, eval_dataset)
    print(f"Evaluation results: {eval_results}")
    
    # Step 5: Test the model with some example predictions
    print("\nStep 5: Testing model with example texts...")
    example_texts = [
        "The company has expressed concerns about increased trade restrictions affecting their supply chain.",
        "New tariffs have been implemented, which could result in higher costs for imported materials.",
        "The free trade agreement has created new opportunities for expanding into international markets.",
        "Global operations were temporarily disrupted due to international conflicts in key regions.",
        "Management is considering relocating production facilities due to favorable domestic incentives."
    ]
    
    predictions = trainer.predict_impact(example_texts)
    
    print("\nExample Predictions:")
    print("--------------------")
    for i, pred in enumerate(predictions, 1):
        print(f"Example {i}: \"{pred['text']}\"")
        print(f"Predicted impact: {pred['predicted_impact']} (confidence: {pred['confidence']:.2f})")
        print("")
    
    # Step 6: Use the model in the impact analyzer workflow
    print("\nStep 6: Generating impact analysis with fine-tuned model...")
    impact = ImpactAnalyzer(topic="protectionism", output_dir=output_dir)
    
    # For real analysis, you would:
    # 1. Pass model predictions to feature engineering
    # 2. Generate visualizations and reports
    
    print("\nModel fine-tuning workflow complete!")
    print(f"Fine-tuned model saved to: {model_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())