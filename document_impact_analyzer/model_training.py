"""
Model training module for topic impact analysis.
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    """Class to train models on topic-specific data"""
    
    def __init__(self, topic_name="Custom Topic", model_name="distilbert-base-uncased", output_dir="./model"):
        """
        Initialize the model trainer with the specified pre-trained model
        
        Args:
            topic_name (str): Name of the topic (for model naming)
            model_name (str): Name of the pre-trained model to use
            output_dir (str): Directory to save the fine-tuned model
        """
        self.topic_name = topic_name
        self.model_name = model_name
        self.output_dir = os.path.join(output_dir, topic_name.lower().replace(" ", "_"))
        self.use_transformer = False
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Try to load HuggingFace transformers, with graceful fallback to sklearn
        try:
            from transformers import (
                AutoTokenizer, 
                AutoModelForSequenceClassification, 
                TrainingArguments, 
                Trainer,
                DataCollatorWithPadding
            )
            from datasets import Dataset
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=3  # Impact categories: negative, neutral, positive
            )
            
            # Data collator for padding
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.use_transformer = True
            print(f"Loaded transformer model {model_name} successfully")
            
            # Keep references to transformer-specific functionality
            self.Dataset = Dataset
            self.TrainingArguments = TrainingArguments
            self.Trainer = Trainer
            
        except Exception as e:
            print(f"Error loading transformer model {model_name}: {e}")
            print("Falling back to sklearn LogisticRegression model")
            self.sklearn_model = LogisticRegression(
                C=1.0, 
                max_iter=1000,
                class_weight='balanced',
                n_jobs=-1
            )
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
    
    def prepare_dataset(self, labeled_data_path):
        """
        Prepare dataset for training
        
        Args:
            labeled_data_path (str): Path to the labeled data CSV file
            
        Returns:
            tuple: Training and evaluation datasets
        """
        # Load labeled data
        df = pd.read_csv(labeled_data_path)
        
        # Convert sentiment labels to numeric labels if necessary
        sentiment_map = {
            "NEGATIVE": 0,
            "NEUTRAL": 1,
            "POSITIVE": 2
        }
        
        # Check if sentiment labels need mapping
        if "sentiment_label" in df.columns:
            if isinstance(df["sentiment_label"].iloc[0], str):
                df["label"] = df["sentiment_label"].map(lambda x: sentiment_map.get(x, 1))  # Default to neutral
            else:
                df["label"] = df["sentiment_label"]
        else:
            # If no sentiment labels, create binary labels based on relevance score
            df["label"] = (df["relevance_score"] > 0.3).astype(int)
        
        # Split the dataset into training and evaluation
        train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
        
        if self.use_transformer:
            # Convert to HuggingFace datasets
            train_dataset = self.Dataset.from_pandas(train_df)
            eval_dataset = self.Dataset.from_pandas(eval_df)
            
            # Tokenize the datasets
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["content"], 
                    padding=False,
                    truncation=True,
                    max_length=512
                )
            
            tokenized_train = train_dataset.map(tokenize_function, batched=True)
            tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
            
            return tokenized_train, tokenized_eval
        else:
            # For sklearn approach, just return the dataframes
            return train_df, eval_df
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics for transformer models
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            dict: Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, eval_dataset, epochs=3, batch_size=16):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            object: Trained model
        """
        if self.use_transformer:
            # Define training arguments for transformer
            training_args = self.TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            # Initialize trainer
            trainer = self.Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics
            )
            
            # Start training
            print(f"Starting transformer model fine-tuning for {self.topic_name}...")
            trainer.train()
            
            # Save the best model
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            return trainer
        else:
            # Train sklearn model
            print(f"Training sklearn LogisticRegression model for {self.topic_name}...")
            
            # Extract text and labels
            X_train = train_dataset["content"].values
            y_train = train_dataset["label"].values
            
            # Create TF-IDF features
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            
            # Train the model
            self.sklearn_model.fit(X_train_tfidf, y_train)
            
            # Save the model and vectorizer
            try:
                import joblib
                joblib.dump(self.sklearn_model, os.path.join(self.output_dir, "sklearn_model.joblib"))
                joblib.dump(self.vectorizer, os.path.join(self.output_dir, "tfidf_vectorizer.joblib"))
                print(f"Saved sklearn model to {self.output_dir}")
            except Exception as e:
                print(f"Error saving sklearn model: {e}")
            
            return self.sklearn_model
    
    def evaluate(self, model, eval_dataset):
        """
        Evaluate the trained model
        
        Args:
            model: Trained model
            eval_dataset: Evaluation dataset
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model...")
        
        if self.use_transformer:
            # Evaluate transformer model
            eval_results = model.evaluate(eval_dataset)
            return eval_results
        else:
            # Evaluate sklearn model
            X_eval = eval_dataset["content"].values
            y_eval = eval_dataset["label"].values
            
            # Create TF-IDF features
            X_eval_tfidf = self.vectorizer.transform(X_eval)
            
            # Make predictions
            y_pred = self.sklearn_model.predict(X_eval_tfidf)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_eval, y_pred, average='weighted'
            )
            acc = accuracy_score(y_eval, y_pred)
            
            eval_results = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            
            print(f"Evaluation results: {eval_results}")
            return eval_results
    
    def predict_impact(self, texts):
        """
        Predict the impact on given texts
        
        Args:
            texts (list): List of text strings to analyze
            
        Returns:
            list: List of prediction results
        """
        if self.use_transformer:
            # Predict with transformer model
            self.model.eval()
            
            # Force CPU for inference to avoid MPS/CUDA issues
            device = "cpu"
            self.model = self.model.to(device)
            
            # Tokenize the texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(predictions, dim=1)
            
            # Format results
            results = []
            for i, (text, pred_class, pred_probs) in enumerate(zip(texts, predicted_classes, predictions)):
                impact_map = {
                    0: "Negative Impact",
                    1: "Neutral Impact",
                    2: "Positive Impact"
                }
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "predicted_impact": impact_map[pred_class.item()],
                    "confidence": pred_probs[pred_class].item(),
                    "probabilities": {
                        impact_map[j]: prob.item() for j, prob in enumerate(pred_probs)
                    }
                })
            
            return results
        else:
            # Predict with sklearn model
            # Create TF-IDF features
            text_features = self.vectorizer.transform(texts)
            
            # Make predictions
            predictions = self.sklearn_model.predict(text_features)
            probabilities = self.sklearn_model.predict_proba(text_features)
            
            # Format results
            results = []
            impact_map = {
                0: "Negative Impact",
                1: "Neutral Impact",
                2: "Positive Impact"
            }
            
            for i, (text, pred_class, pred_probs) in enumerate(zip(texts, predictions, probabilities)):
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "predicted_impact": impact_map[pred_class],
                    "confidence": pred_probs[pred_class],
                    "probabilities": {
                        impact_map[j]: prob for j, prob in enumerate(pred_probs)
                    }
                })
            
            return results

def train_model(topic_name="Custom Topic", labeled_data_path=None, model_name="distilbert-base-uncased", output_dir="./model"):
    """
    Helper function to train a model on labeled data
    
    Args:
        topic_name (str): Name of the topic for the model
        labeled_data_path (str): Path to the labeled data CSV file
        model_name (str): Name of the pre-trained model to use
        output_dir (str): Directory to save the fine-tuned model
        
    Returns:
        tuple: Trainer and evaluation results
    """
    if labeled_data_path is None or not os.path.exists(labeled_data_path):
        print(f"Error: Labeled data file not found at {labeled_data_path}")
        return None, None
    
    # Initialize model trainer
    trainer = ModelTrainer(topic_name=topic_name, model_name=model_name, output_dir=output_dir)
    
    # Prepare dataset
    train_dataset, eval_dataset = trainer.prepare_dataset(labeled_data_path)
    
    # Train the model
    trained_model = trainer.train(train_dataset, eval_dataset)
    
    # Evaluate the model
    eval_results = trainer.evaluate(trained_model, eval_dataset)
    print(f"Evaluation results: {eval_results}")
    
    # Test some predictions
    sample_texts = [
        "The company reported significant challenges related to new regulations in this area.",
        "We are adapting our strategy to address these emerging issues in our industry.",
        "This development represents a positive opportunity for our business growth."
    ]
    
    predictions = trainer.predict_impact(sample_texts)
    for pred in predictions:
        print(f"Text: {pred['text']}")
        print(f"Predicted impact: {pred['predicted_impact']} (confidence: {pred['confidence']:.2f})")
        print("---")
    
    return trainer, eval_results