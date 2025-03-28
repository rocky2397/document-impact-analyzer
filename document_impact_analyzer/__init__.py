"""
Document Impact Analyzer

A tool for analyzing the impact of specific topics in corporate documents and reports.
"""

__version__ = "0.1.0"

from document_impact_analyzer.text_analysis import TextAnalyzer
from document_impact_analyzer.data_extraction import DataExtractor
from document_impact_analyzer.impact_analyzer import ImpactAnalyzer
from document_impact_analyzer.model_training import ModelTrainer, train_model

__all__ = ["TextAnalyzer", "DataExtractor", "ImpactAnalyzer", "ModelTrainer", "train_model"]