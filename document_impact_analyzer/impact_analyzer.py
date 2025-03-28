"""
Impact analyzer module for evaluating topic impacts across documents.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json

from document_impact_analyzer.config import default_config

class ImpactAnalyzer:
    """
    Main class to analyze the impact of specific topics on companies
    """
    
    def __init__(self, topic=None, output_dir="./output", custom_keywords=None, scoring_config=None):
        """
        Initialize the ImpactAnalyzer
        
        Args:
            topic (str, optional): Topic to analyze (must exist in config)
            output_dir (str): Directory to store output files
            custom_keywords (list, optional): Custom list of keywords if not using a predefined topic
            scoring_config (dict, optional): Custom scoring configuration
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the topic
        self.topic = topic or default_config.DEFAULT_TOPIC
        self.custom_keywords = custom_keywords
        
        if custom_keywords:
            self.topic_name = "Custom Topic"
            self.topic_description = "Custom topic analysis"
        elif topic in default_config.TOPICS:
            self.topic_name = default_config.TOPICS[topic]["name"]
            self.topic_description = default_config.TOPICS[topic]["description"]
        else:
            # Default to the default topic
            self.topic = default_config.DEFAULT_TOPIC
            self.topic_name = default_config.TOPICS[self.topic]["name"]
            self.topic_description = default_config.TOPICS[self.topic]["description"]
        
        # Set up scoring configuration
        self.scoring_config = scoring_config or default_config.IMPACT_SCORING
        
        # Try to import visualization libraries with graceful fallback
        self.has_plotting = False
        self.has_wordcloud = False
        self.has_plotly = False
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self.has_plotting = True
            print("Matplotlib and Seaborn loaded successfully")
        except Exception as e:
            print(f"Visualization libraries not available: {e}")
        
        try:
            from wordcloud import WordCloud
            self.WordCloud = WordCloud
            self.has_wordcloud = True
            print("WordCloud loaded successfully")
        except Exception as e:
            print(f"WordCloud not available: {e}")
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            self.px = px
            self.go = go
            self.has_plotly = True
            print("Plotly loaded successfully")
        except Exception as e:
            print(f"Plotly not available: {e}")
        
        # Try to initialize summarization pipeline
        self.has_summarizer = False
        try:
            from transformers import pipeline
            self.summarizer = pipeline("summarization")
            self.has_summarizer = True
            print("Summarization pipeline loaded successfully")
        except Exception as e:
            print(f"Summarization not available: {e}")
        
        # Data storage
        self.documents = None
        self.labeled_data = None
        self.feature_df = None
        self.company_scores = None
    
    def engineer_features(self, labeled_data=None):
        """
        Engineer features from labeled data to analyze impact
        
        Args:
            labeled_data: Either DataFrame or path to CSV with labeled data
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        if labeled_data is not None:
            self.labeled_data = labeled_data
            
        if self.labeled_data is None:
            print("No labeled data available.")
            return None
        
        # Load the labeled data if it's a string path
        if isinstance(self.labeled_data, str):
            self.labeled_data = pd.read_csv(self.labeled_data)
        
        # Create a feature dataframe
        feature_data = []
        
        # Group by company and year
        grouped = self.labeled_data.groupby(['company', 'year'])
        
        for (company, year), group in grouped:
            # Count mentions
            mention_count = len(group)
            
            # Calculate average relevance score
            avg_relevance = group['relevance_score'].mean()
            
            # Calculate sentiment distribution
            if 'sentiment_label' in group.columns:
                sentiment_counts = group['sentiment_label'].value_counts(normalize=True).to_dict()
                negative_ratio = sentiment_counts.get('NEGATIVE', 0)
                positive_ratio = sentiment_counts.get('POSITIVE', 0)
                neutral_ratio = sentiment_counts.get('NEUTRAL', 0)
            else:
                negative_ratio = positive_ratio = neutral_ratio = 0
            
            # Calculate sentiment score (-1 to 1 scale)
            sentiment_score = (positive_ratio - negative_ratio)
            
            # Calculate weighted impact score using config weights
            relevance_weight = self.scoring_config.get("relevance_weight", 0.6)
            sentiment_weight = self.scoring_config.get("sentiment_weight", 0.4)
            impact_score = (avg_relevance * relevance_weight) * (1 + sentiment_score * sentiment_weight)
            
            # Create feature vector
            features = {
                'company': company,
                'year': year,
                'mention_count': mention_count,
                'avg_relevance': avg_relevance,
                'sentiment_score': sentiment_score,
                'negative_ratio': negative_ratio,
                'neutral_ratio': neutral_ratio,
                'positive_ratio': positive_ratio,
                'impact_score': impact_score,
            }
            
            # Generate summary if summarizer is available
            if self.has_summarizer:
                all_text = " ".join(group['content'].head(5))  # Take first 5 chunks to avoid token limits
                try:
                    summary = self.summarizer(all_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                    features['summary'] = summary
                except Exception as e:
                    print(f"Error generating summary for {company} {year}: {e}")
                    features['summary'] = "Summary not available"
            
            feature_data.append(features)
        
        # Create feature dataframe
        self.feature_df = pd.DataFrame(feature_data)
        
        # Calculate company-level scores
        self.calculate_company_scores()
        
        # Calculate feature importance for prediction
        self.calculate_feature_importance()
        
        # Save feature data
        feature_file = os.path.join(self.output_dir, f"{self.topic_name.lower().replace(' ', '_')}_features.csv")
        self.feature_df.to_csv(feature_file, index=False)
        print(f"Feature data saved to {feature_file}")
        
        return self.feature_df
    
    def calculate_company_scores(self):
        """
        Calculate overall impact scores for each company
        
        Returns:
            pd.DataFrame: Dataframe with company scores
        """
        if self.feature_df is None:
            print("No feature data available. Run feature engineering first.")
            return None
        
        # Calculate overall scores for each company
        company_data = []
        
        # Get thresholds from config
        low_risk_threshold = self.scoring_config.get("low_risk_threshold", 0.5)
        high_risk_threshold = self.scoring_config.get("high_risk_threshold", 1.5)
        
        for company in self.feature_df['company'].unique():
            company_features = self.feature_df[self.feature_df['company'] == company]
            
            # Calculate overall metrics
            avg_impact = company_features['impact_score'].mean()
            avg_sentiment = company_features['sentiment_score'].mean()
            total_mentions = company_features['mention_count'].sum()
            
            # Calculate trend (year-over-year change)
            if len(company_features) > 1:
                company_features = company_features.sort_values('year')
                impact_values = company_features['impact_score'].values
                trend = np.mean(np.diff(impact_values))
            else:
                trend = 0
            
            # Calculate vulnerability score
            mention_weight = self.scoring_config.get("mention_weight", 0.2)
            vulnerability = avg_impact * (1 + abs(avg_sentiment)) * np.log1p(total_mentions * mention_weight)
            
            # Determine risk category
            if vulnerability < low_risk_threshold:
                risk_category = "Low Risk"
            elif vulnerability < high_risk_threshold:
                risk_category = "Moderate Risk"
            else:
                risk_category = "High Risk"
            
            company_data.append({
                'company': company,
                'avg_impact_score': avg_impact,
                'avg_sentiment': avg_sentiment,
                'total_mentions': total_mentions,
                'trend': trend,
                'vulnerability_score': vulnerability,
                'risk_category': risk_category
            })
        
        self.company_scores = pd.DataFrame(company_data)
        
        # Save company scores
        scores_file = os.path.join(self.output_dir, f"company_{self.topic_name.lower().replace(' ', '_')}_scores.csv")
        self.company_scores.to_csv(scores_file, index=False)
        print(f"Company scores saved to {scores_file}")
        
        return self.company_scores
    
    def calculate_feature_importance(self):
        """
        Calculate importance of different features in predicting impact
        
        Returns:
            pd.DataFrame: Dataframe with feature importance scores
        """
        if self.labeled_data is None:
            print("No labeled data available.")
            return None
        
        # Extract text from labeled data
        texts = self.labeled_data['content'].values
        
        # Use TF-IDF to identify important terms
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            
            # Get feature names
            feature_names = tfidf.get_feature_names_out()
            
            # Calculate importance scores (mean TF-IDF across documents)
            importance_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
            
            # Create feature importance dataframe
            importance_data = {
                'term': feature_names,
                'importance': importance_scores
            }
            
            self.feature_importance = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
            
            # Save feature importance
            importance_file = os.path.join(self.output_dir, f"{self.topic_name.lower().replace(' ', '_')}_term_importance.csv")
            self.feature_importance.to_csv(importance_file, index=False)
            print(f"Feature importance saved to {importance_file}")
            
            return self.feature_importance
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return None
    
    def generate_visualizations(self):
        """
        Generate visualizations of the analysis results
        
        Returns:
            None
        """
        if self.feature_df is None or self.company_scores is None:
            print("No data available for visualization. Run feature engineering first.")
            return
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate visualizations based on available libraries
        if self.has_plotly:
            self._generate_plotly_visualizations(viz_dir)
        
        if self.has_plotting and self.has_wordcloud and hasattr(self, 'feature_importance'):
            self._generate_wordcloud(viz_dir)
        
        print(f"Visualizations saved to {viz_dir}")
    
    def _generate_plotly_visualizations(self, viz_dir):
        """Generate visualizations using Plotly"""
        # Use color map from config
        color_map = default_config.VISUALIZATION.get("color_map", {
            'Low Risk': 'green',
            'Moderate Risk': 'orange',
            'High Risk': 'red'
        })
        
        # 1. Company Vulnerability Comparison
        fig = self.px.bar(
            self.company_scores, 
            x='company', 
            y='vulnerability_score',
            color='risk_category',
            title=f'Company Vulnerability to {self.topic_name}',
            labels={'vulnerability_score': 'Vulnerability Score', 'company': 'Company'},
            color_discrete_map=color_map
        )
        fig.write_html(os.path.join(viz_dir, f"{self.topic_name.lower().replace(' ', '_')}_company_vulnerability.html"))
        
        # 2. Mentions Over Time
        time_data = self.feature_df.pivot(index='year', columns='company', values='mention_count')
        fig = self.px.line(
            time_data, 
            title=f'{self.topic_name} Mentions by Company Over Time',
            labels={'value': 'Number of Mentions', 'year': 'Year'}
        )
        fig.write_html(os.path.join(viz_dir, f"{self.topic_name.lower().replace(' ', '_')}_mentions_over_time.html"))
        
        # 3. Impact Score Heatmap
        impact_data = self.feature_df.pivot(index='company', columns='year', values='impact_score')
        fig = self.px.imshow(
            impact_data,
            title=f'{self.topic_name} Impact Score by Company and Year',
            labels={'x': 'Year', 'y': 'Company', 'color': 'Impact Score'},
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.write_html(os.path.join(viz_dir, f"{self.topic_name.lower().replace(' ', '_')}_impact_heatmap.html"))
        
        # 4. Sentiment Distribution
        sentiment_data = pd.melt(
            self.feature_df, 
            id_vars=['company', 'year'],
            value_vars=['negative_ratio', 'neutral_ratio', 'positive_ratio'],
            var_name='sentiment',
            value_name='ratio'
        )
        fig = self.px.bar(
            sentiment_data,
            x='company',
            y='ratio',
            color='sentiment',
            facet_col='year',
            title=f'Sentiment Distribution by Company and Year for {self.topic_name}',
            labels={'ratio': 'Proportion', 'company': 'Company'},
            color_discrete_map={
                'negative_ratio': 'red',
                'neutral_ratio': 'gray',
                'positive_ratio': 'green'
            }
        )
        fig.write_html(os.path.join(viz_dir, f"{self.topic_name.lower().replace(' ', '_')}_sentiment_distribution.html"))
    
    def _generate_wordcloud(self, viz_dir):
        """Generate wordcloud visualization"""
        # Create term frequency dictionary for wordcloud
        word_freq = dict(zip(self.feature_importance['term'], self.feature_importance['importance']))
        
        # Generate wordcloud
        wordcloud = self.WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        # Save wordcloud
        self.plt.figure(figsize=(10, 5))
        self.plt.imshow(wordcloud, interpolation='bilinear')
        self.plt.axis('off')
        self.plt.tight_layout()
        self.plt.savefig(os.path.join(viz_dir, f"{self.topic_name.lower().replace(' ', '_')}_terms_wordcloud.png"), dpi=300)
        self.plt.close()
    
    def generate_company_report(self, company):
        """
        Generate a detailed report for a specific company
        
        Args:
            company (str): Name of the company
            
        Returns:
            str: Report text
        """
        if self.feature_df is None or self.company_scores is None:
            print("No data available for report generation. Run feature engineering first.")
            return "No data available for report generation."
        
        # Get company data
        company_features = self.feature_df[self.feature_df['company'] == company]
        company_score = self.company_scores[self.company_scores['company'] == company].iloc[0]
        
        # Generate report
        report = f"# {self.topic_name} Impact Analysis for {company}\n\n"
        
        report += "## Summary\n\n"
        report += f"Vulnerability Score: {company_score['vulnerability_score']:.2f} ({company_score['risk_category']})\n\n"
        report += f"Average Impact Score: {company_score['avg_impact_score']:.2f}\n\n"
        report += f"Total {self.topic_name} Mentions: {company_score['total_mentions']}\n\n"
        
        if company_score['trend'] > 0:
            trend_desc = "increasing"
        elif company_score['trend'] < 0:
            trend_desc = "decreasing"
        else:
            trend_desc = "stable"
        report += f"Year-over-Year Trend: {trend_desc} ({company_score['trend']:.2f})\n\n"
        
        report += "## Yearly Analysis\n\n"
        for _, row in company_features.sort_values('year').iterrows():
            report += f"### {row['year']}\n\n"
            report += f"Impact Score: {row['impact_score']:.2f}\n\n"
            report += f"Mentions: {row['mention_count']}\n\n"
            
            if 'summary' in row:
                report += f"Key Insights: {row['summary']}\n\n"
        
        report += "## Recommendation\n\n"
        
        # Get recommendations from config
        high_risk_recs = default_config.RECOMMENDATIONS.get("high_risk", [])
        moderate_risk_recs = default_config.RECOMMENDATIONS.get("moderate_risk", [])
        low_risk_recs = default_config.RECOMMENDATIONS.get("low_risk", [])
        
        if company_score['vulnerability_score'] > self.scoring_config.get("high_risk_threshold", 1.5):
            report += f"The company has a high vulnerability to {self.topic_name.lower()}. It is recommended to:\n\n"
            for i, rec in enumerate(high_risk_recs, 1):
                report += f"{i}. {rec}\n"
        elif company_score['vulnerability_score'] > self.scoring_config.get("low_risk_threshold", 0.5):
            report += f"The company has a moderate vulnerability to {self.topic_name.lower()}. It is recommended to:\n\n"
            for i, rec in enumerate(moderate_risk_recs, 1):
                report += f"{i}. {rec}\n"
        else:
            report += f"The company has a low vulnerability to {self.topic_name.lower()}. It is recommended to:\n\n"
            for i, rec in enumerate(low_risk_recs, 1):
                report += f"{i}. {rec}\n"
        
        # Save report
        report_file = os.path.join(self.output_dir, f"{company}_{self.topic_name.lower().replace(' ', '_')}_report.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Company report saved to {report_file}")
        return report