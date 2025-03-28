"""
Basic unit tests for document_impact_analyzer package.
"""

import os
import unittest
import pandas as pd
from document_impact_analyzer import TextAnalyzer, DataExtractor, ImpactAnalyzer

class TestTextAnalyzer(unittest.TestCase):
    """Test the TextAnalyzer class functionality"""
    
    def test_initialization(self):
        """Test that TextAnalyzer initializes correctly with default topic"""
        analyzer = TextAnalyzer()
        self.assertIsNotNone(analyzer.keywords)
        self.assertGreater(len(analyzer.keywords), 0)
        
    def test_custom_keywords(self):
        """Test initialization with custom keywords"""
        custom_keywords = ["test", "custom", "keywords"]
        analyzer = TextAnalyzer(custom_keywords=custom_keywords)
        self.assertEqual(set(analyzer.keywords), set(custom_keywords))
        
    def test_climate_change_topic(self):
        """Test initialization with climate_change topic"""
        analyzer = TextAnalyzer(topic="climate_change")
        self.assertIsNotNone(analyzer.keywords)
        self.assertGreater(len(analyzer.keywords), 0)
        self.assertIn("climate change", analyzer.keywords)

class TestImpactAnalyzer(unittest.TestCase):
    """Test the ImpactAnalyzer class functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a small test DataFrame
        self.test_data = pd.DataFrame({
            'content': [
                "Company is affected by tariffs and trade barriers.",
                "This text has nothing relevant to protectionism.",
                "Export duties have significantly affected our business."
            ],
            'company': ['TestCo', 'TestCo', 'AnotherCo'],
            'year': [2022, 2022, 2023],
            'relevance_score': [0.8, 0.1, 0.9],
            'sentiment_label': ['NEGATIVE', 'NEUTRAL', 'NEGATIVE'],
            'sentiment_score': [0.9, 0.5, 0.8]
        })
        
        # Create a temporary file for testing
        self.temp_csv = 'temp_test_data.csv'
        self.test_data.to_csv(self.temp_csv, index=False)
        
        # Initialize analyzer with temp output directory
        self.output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.impact = ImpactAnalyzer(topic="protectionism", output_dir=self.output_dir)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)
    
    def test_feature_engineering(self):
        """Test feature engineering on test data"""
        features = self.impact.engineer_features(self.temp_csv)
        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)
        
        # Check if company scores were calculated
        self.assertIsNotNone(self.impact.company_scores)
        self.assertIn('company', self.impact.company_scores.columns)
        
if __name__ == '__main__':
    unittest.main()