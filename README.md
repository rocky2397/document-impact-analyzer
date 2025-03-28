# Document Impact Analyzer

A flexible tool for analyzing the impact of specific topics in corporate documents and annual reports.

## Overview

Document Impact Analyzer is a Python package that helps you identify and measure the impact of any topic (like protectionism, climate change, etc.) in corporate documents such as annual reports. It combines natural language processing, sentiment analysis, and data visualization to provide insights into how companies are affected by or responding to specific topics.

Originally developed to analyze protectionism impacts, the tool has been generalized to work with any user-defined topic and keywords.

## Features

- **Flexible topic definition**: Analyze any topic by defining custom keywords or using pre-configured topics
- **Document extraction**: Extract text from PDFs, TXT files, and CSV data
- **Topic relevance detection**: Identify content related to your chosen topic
- **Sentiment analysis**: Determine the sentiment of topic mentions
- **Impact scoring**: Calculate topic impact and vulnerability scores for companies
- **Trend analysis**: Track changes in topic mentions and impact over time
- **Rich visualizations**: Generate interactive charts and word clouds
- **Detailed reporting**: Create company-specific analysis reports with recommendations
- **Model fine-tuning**: Train custom models on your labeled data for improved analysis

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/yourusername/document-impact-analyzer.git
```

For the full functionality including vector search and advanced NLP features:

```bash
pip install git+https://github.com/yourusername/document-impact-analyzer.git#egg=document-impact-analyzer[full]
```

Alternatively, clone the repository and install locally:

```bash
git clone https://github.com/yourusername/document-impact-analyzer.git
cd document-impact-analyzer
pip install -e .
```

## Quick Start

### Basic Usage

```python
from document_impact_analyzer import TextAnalyzer, DataExtractor, ImpactAnalyzer

# Extract documents from PDF files
extractor = DataExtractor(data_dir="./data")
documents = extractor.extract_documents(companies=["BP", "CHEVRON"], years=["2021", "2022", "2023"])

# Analyze text for protectionism (the default topic)
analyzer = TextAnalyzer()
relevant_docs = analyzer.identify_relevant_content(documents)
analyzed_docs = analyzer.analyze_sentiment(relevant_docs)

# Export the labeled data
labeled_data = analyzer.export_labeled_data(analyzed_docs, output_path="labeled_data.csv")

# Generate impact analysis
impact = ImpactAnalyzer(output_dir="./output")
impact.engineer_features(labeled_data)
impact.generate_visualizations()

# Generate a company report
report = impact.generate_company_report("BP")
```

### Custom Topic Analysis

```python
# Define custom keywords for climate change analysis
climate_keywords = [
    "climate change", "global warming", "carbon emissions", 
    "renewable energy", "greenhouse gas", "sustainability"
]

# Analyze text with custom keywords
analyzer = TextAnalyzer(custom_keywords=climate_keywords)
relevant_docs = analyzer.identify_relevant_content(documents)

# Or use a predefined topic
analyzer = TextAnalyzer(topic="climate_change")
```

### Model Fine-Tuning

```python
from document_impact_analyzer import ModelTrainer

# Initialize a model trainer for your topic
trainer = ModelTrainer(topic_name="Protectionism", output_dir="./models")

# Prepare dataset from your labeled data
train_dataset, eval_dataset = trainer.prepare_dataset("labeled_data.csv")

# Train the model
trained_model = trainer.train(train_dataset, eval_dataset, epochs=3)

# Evaluate performance
eval_results = trainer.evaluate(trained_model, eval_dataset)

# Make predictions on new text
example_texts = [
    "Company reported increased tariffs affecting their supply chain.",
    "New free trade agreement provides opportunities for expansion."
]
predictions = trainer.predict_impact(example_texts)
```

## Testing if it works

For a quick test to see if the package is working correctly, you can run the included test script:

```bash
# From the root directory of the package
python -m examples.quick_test
```

This will run a simple analysis using the sample data provided and generate basic output to verify functionality.

## Data Handling

### Working with PDF files

The original analysis was performed on PDF annual reports from energy companies. The package supports multiple methods for PDF text extraction:

1. **Built-in extraction**: The `DataExtractor` class has built-in support for PDF extraction using PyPDF2 with fallbacks to pdfplumber if available.

2. **NLM-Ingestor (recommended for complex PDFs)**: For better extraction from complex, multi-column PDFs, you can use [NLM-Ingestor](https://github.com/NIHOPA/NLM-Ingestor) which was used in the original analysis:

```bash
# Install NLM-Ingestor
pip install git+https://github.com/NIHOPA/NLM-Ingestor.git

# Extract text from PDFs
python -m nlm_ingestor.ingestor --input ./data --output ./extracted_text
```

3. **Pre-processed data**: For convenience, this package includes compressed versions of pre-processed text extractions from annual reports in the `examples/data` directory.

### Sample Data

The package includes compressed sample data extracted from annual reports of several energy companies (BP, CHEVRON, EON, RWE, NEXTERA) for years 2021-2023. This data is provided in CSV format to make it easier to test and use the package without requiring large PDF files.

To use your own data, you can:
1. Process PDF files directly using the `DataExtractor`
2. Use external tools like NLM-Ingestor for better extraction from complex PDFs
3. Prepare your own CSV files following the sample format

## Data Structure

The tool expects corporate documents to be organized in a structure like:

```
data/
  ├── COMPANY1/
  │   ├── COMPANY1_2021.pdf
  │   └── COMPANY1_2022.pdf
  └── COMPANY2/
      ├── COMPANY2_2021.pdf
      └── COMPANY2_2022.pdf
```

Alternatively, you can use a flat structure with company and year in the filename:

```
data/
  ├── COMPANY1_2021.pdf
  ├── COMPANY1_2022.pdf
  ├── COMPANY2_2021.pdf
  └── COMPANY2_2022.pdf
```

## Configuring Topics

You can define custom topics by editing the `config/default_config.py` file or by passing custom keywords at runtime. 

The default configuration includes these topics:
- Protectionism 
- Climate Change
- Digital Transformation

## Example Output

The tool generates several types of outputs:

1. **CSV files** with labeled data and feature analyses
2. **Interactive visualizations** showing impact scores, trends, and comparisons
3. **Company reports** with detailed analysis and recommendations
4. **Fine-tuned models** trained on your labeled data

## Complete Workflow Examples

The package includes several example scripts demonstrating the complete workflow:

- `example_analysis.py` - Basic analysis of corporate documents
- `example_climate_analysis.py` - Analysis focused on climate change
- `example_model_finetuning.py` - Complete workflow for fine-tuning a custom model
- `quick_test.py` - Simple test script to verify the package is working

Run these examples from the command line:

```bash
python -m examples.example_analysis
python -m examples.example_model_finetuning
python -m examples.quick_test
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.