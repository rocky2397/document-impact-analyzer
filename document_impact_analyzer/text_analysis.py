"""
Text analysis module for identifying and analyzing specific topics in documents.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from document_impact_analyzer.config import default_config

class TextAnalyzer:
    """
    Class to identify and label text chunks related to specific topics.
    This is a generalized version that can work with any topic keywords.
    """
    
    def __init__(self, topic=None, custom_keywords=None, vector_store_dir='./chroma_db'):
        """
        Initialize the TextAnalyzer with embeddings and relevant keywords
        
        Args:
            topic (str, optional): Topic identifier from config (e.g., "protectionism")
            custom_keywords (list, optional): Custom list of keywords if not using a predefined topic
            vector_store_dir (str): Path to store the vector database
        """
        self.use_vector_store = False
        self.topic_name = "Custom Topic"
        self.topic_description = "Custom topic analysis"
        
        # Set the keywords based on the topic or custom list
        if custom_keywords:
            self.keywords = custom_keywords
            if topic:
                print(f"Warning: Both topic and custom_keywords provided. Using custom_keywords.")
        elif topic and topic in default_config.TOPICS:
            self.topic_name = default_config.TOPICS[topic]["name"]
            self.topic_description = default_config.TOPICS[topic]["description"]
            self.keywords = default_config.TOPICS[topic]["keywords"]
        elif topic:
            raise ValueError(f"Topic '{topic}' not found in configuration. Available topics: {list(default_config.TOPICS.keys())}")
        else:
            # Default to protectionism if no topic is specified
            default_topic = default_config.DEFAULT_TOPIC
            self.topic_name = default_config.TOPICS[default_topic]["name"]
            self.topic_description = default_config.TOPICS[default_topic]["description"]
            self.keywords = default_config.TOPICS[default_topic]["keywords"]
            print(f"No topic specified. Using default: {default_topic}")
        
        # Try to initialize vector store, but with graceful fallback
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma
            
            self.embeddings = HuggingFaceEmbeddings()
            
            # Initialize vector store for semantic search
            self.vector_store = Chroma(
                collection_name=f"{self.topic_name.lower().replace(' ', '_')}_analysis",
                embedding_function=self.embeddings,
                persist_directory=vector_store_dir
            )
            self.use_vector_store = True
            print(f"Vector store initialized successfully for {self.topic_name}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            print("Using TF-IDF instead of vector store for document similarity")
            self.vector_store = None
            self.embeddings = None
        
        # Try to load the sentiment pipeline
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            print("Sentiment analysis pipeline loaded successfully")
        except Exception as e:
            print(f"Error loading sentiment pipeline: {e}")
            self.sentiment_pipeline = None
    
    def add_documents_to_vectorstore(self, documents):
        """
        Add documents to the vector store for later retrieval
        
        Args:
            documents (list): List of LangChain Document objects
            
        Returns:
            None
        """
        if not self.use_vector_store:
            print("Vector store not available, skipping document addition")
            return
            
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
        print(f"Added {len(documents)} documents to the vector store")
    
    def identify_relevant_content(self, documents, threshold=0.15):
        """
        Identify content related to the topic using keyword matching and TF-IDF
        
        Args:
            documents (list): List of LangChain Document objects
            threshold (float): Similarity threshold to consider a chunk relevant
            
        Returns:
            list: Filtered list of documents related to the topic
        """
        if not documents:
            print("No documents provided for analysis")
            return []
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Use TF-IDF to create document vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        document_vectors = vectorizer.fit_transform(texts)
        
        # Create a vector for topic keywords
        keywords_text = " ".join(self.keywords)
        keyword_vector = vectorizer.transform([keywords_text])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(document_vectors, keyword_vector).flatten()
        
        # Filter documents based on similarity threshold
        relevant_docs = []
        for i, (doc, score) in enumerate(zip(documents, similarity_scores)):
            if score > threshold:
                # Add relevance score to metadata
                doc.metadata["relevance_score"] = float(score)
                relevant_docs.append(doc)
        
        print(f"Identified {len(relevant_docs)} documents relevant to {self.topic_name} out of {len(documents)} total")
        return relevant_docs
    
    def analyze_sentiment(self, documents):
        """
        Analyze sentiment of documents
        
        Args:
            documents (list): List of LangChain Document objects
            
        Returns:
            list: Documents with sentiment analysis metadata added
        """
        if not self.sentiment_pipeline:
            print("Sentiment pipeline not available, assigning neutral sentiment")
            for doc in documents:
                doc.metadata["sentiment_label"] = "NEUTRAL"
                doc.metadata["sentiment_score"] = 0.5
            return documents
        
        for doc in documents:
            try:
                # Limit text length for the sentiment pipeline
                text = doc.page_content[:512]  # Truncate to avoid token limit issues
                sentiment_result = self.sentiment_pipeline(text)[0]
                
                # Add sentiment metadata
                doc.metadata["sentiment_label"] = sentiment_result["label"]
                doc.metadata["sentiment_score"] = float(sentiment_result["score"])
            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
                doc.metadata["sentiment_label"] = "NEUTRAL"
                doc.metadata["sentiment_score"] = 0.5
        
        return documents
    
    def semantic_search(self, query, k=5):
        """
        Perform semantic search for topic-related content
        
        Args:
            query (str): Query string to search for
            k (int): Number of results to return
            
        Returns:
            list: Relevant documents
        """
        if not self.use_vector_store:
            print("Vector store not available, cannot perform semantic search")
            return []
            
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def export_labeled_data(self, documents, output_path=None):
        """
        Export labeled data to CSV for further analysis
        
        Args:
            documents (list): List of LangChain Document objects
            output_path (str, optional): Path to save the CSV file
            
        Returns:
            pd.DataFrame: DataFrame of labeled data
        """
        data = []
        
        for doc in documents:
            data.append({
                "content": doc.page_content,
                "company": doc.metadata.get("source", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "type": doc.metadata.get("type", "text"),
                "relevance_score": doc.metadata.get("relevance_score", 0.0),
                "sentiment_label": doc.metadata.get("sentiment_label", "NEUTRAL"),
                "sentiment_score": doc.metadata.get("sentiment_score", 0.5)
            })
        
        df = pd.DataFrame(data)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Exported {len(data)} labeled documents to {output_path}")
        
        return df