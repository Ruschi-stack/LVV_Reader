# LVV (Lestungsverzeichnis-Vorbemerkungen) Reader - German Contract Analysis Model Evaluation

## Table of Contents
- [LVV (Lestungsverzeichnis-Vorbemerkungen) Reader - German Contract Analysis Model Evaluation](#lvv-lestungsverzeichnis-vorbemerkungen-reader---german-contract-analysis-model-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Vorwort](#vorwort)
  - [Project Overview](#project-overview)
  - [Project Goals](#project-goals)
  - [Models Tested](#models-tested)
  - [Evaluation Approach](#evaluation-approach)
  - [Model Comparison](#model-comparison)
    - [deepset/gelectra-base-germanquad](#deepsetgelectra-base-germanquad)
    - [dbmdz/bert-base-german-europeana-cased](#dbmdzbert-base-german-europeana-cased)
    - [GermanT5/t5-efficient-gc4-all-german-small-el32](#germant5t5-efficient-gc4-all-german-small-el32)
    - [Comparison Summary](#comparison-summary)
  - [Extraction Improvement Attempts](#extraction-improvement-attempts)
    - [Pattern Matching Improvements](#pattern-matching-improvements)
    - [Case-Insensitive Search](#case-insensitive-search)
    - [Contextual Search](#contextual-search)
    - [Question Refinement](#question-refinement)
  - [Current Findings](#current-findings)
  - [Future Directions](#future-directions)
  - [Technical Implementation](#technical-implementation)
  - [Usage](#usage)
  - [Conclusion](#conclusion)
  - [Cliffhanger: RAG-Based Solution](#cliffhanger-rag-based-solution)

## Vorwort

The unstated aim of this repo is to display some of the skills I gathered on my learning journey; I used claude 3.5 for assisting and debugging, and a PC without a dedicated GPU or NPU. The "input" folder, where the German contract used in this experiment was stored, is empty in this public repo for privacy reasons.

## Project Overview

This project was initially conceived as a prototype for extracting specific information from a german construction contract (a kind of template) in german language without using a SaaS (ChatGPT & Co.) or expensive hardware (dedicated GPU or NPU). However, it has evolved into a systematic evaluation of small German language models for contract analysis tasks.

## Project Goals

The primary goal of this project is to test and evaluate the performance of various (3 was enough to test my patience) small German language models that have been fine-tuned for German language understanding. I aim to:

1. Compare the effectiveness of different models in extracting specific information from German construction contracts
2. Identify which models perform best for this specialized task
3. Document the strengths and limitations of each model
4. Provide insights for future development of contract analysis tools

## Models Tested

Currently, we have tested the following models:

- **deepset/gelectra-base-germanquad**: A German language model fine-tuned for question answering
- **dbmdz/bert-base-german-europeana-cased**: A German language model based on BERT architecture
- **GermanT5/t5-efficient-gc4-all-german-small-el32**: A German T5 model optimized for efficiency. An extra caviat: BERT and ELECTRA are encoders, while T5 is a encoder-decoder

## Evaluation Approach

Initially, the project planned to use F1 score as a quantitative metric for model evaluation. However, after initial testing, it became apparent that the models' performance was significantly below expectations for this specialized task. The answers extracted were often far from correct, making a quantitative F1 score measurement an overkill.

Instead, I took taking a "qualitative approach" to evaluation, focusing on:

1. **Pattern Recognition**: How well does the model identify relevant sections in the contract?
2. **Context Understanding**: Does the model understand the context of the information it's extracting?
3. **Accuracy**: How often does the model extract the correct information?

"Lange Rede kurzer Sinn": I just looked at the results to see if they selected the correct value for the retentions. 

## Model Comparison

### deepset/gelectra-base-germanquad
- Successfully extracted Bauwasser (0.4 v.H.) and Baustrom (0.2 v.H.) percentages
- Failed to extract Bauwesenversicherung percentage (returned [CLS])
- Failed to extract Sanitäre Anlagen information (returned [SEP])
- Failed to extract Bauschild Euro amount (returned full context with [CLS] and [SEP] markers)

### dbmdz/bert-base-german-europeana-cased
- Successfully extracted Bauwasser (0.4 v.H.) and Baustrom (0.2 v.H.) percentages
- Failed to extract Bauwesenversicherung percentage (returned irrelevant text about Betriebshaftpflichtversicherung)
- Failed to extract Sanitäre Anlagen information (returned empty string)
- Failed to extract Bauschild Euro amount (returned irrelevant text about Betriebshaftpflichtversicherung)

### GermanT5/t5-efficient-gc4-all-german-small-el32
- Successfully extracted Bauwasser (0.4 v.H.) and Baustrom (0.2 v.H.) percentages
- Failed to extract Bauwesenversicherung percentage (returned empty string)
- Failed to extract Sanitäre Anlagen information (returned empty string)
- Failed to extract Bauschild Euro amount (returned empty string)

### Comparison Summary
All three models showed similar limitations, but with different failure modes:
- GermanT5 returns empty strings when it can't find information
- GELECTRA returns token markers ([CLS], [SEP]) and sometimes full context
- BERT returns either empty strings or irrelevant text from the document

All models extracted the Bauwasser and Baustrom percentages with the correct German notation (v.H.), but the wrong percentage (both should have been 0.2 v.H). However, they all failed on the more complex fields, suggesting that the challenge lies in the task complexity rather than the specific model architecture.

## Extraction Improvement Attempts

I have made several attempts to improve the extraction accuracy through various techniques:

### Pattern Matching Improvements
- Implemented regex patterns for specific terms like "bauwasser" and "baustrom"
- Added flexible whitespace matching in patterns
- Created patterns to capture numerical values with various formats (e.g., "0,4%", "0.4%", "0,4 v.H.")
- Attempted to match Euro amounts with different currency symbol positions

### Case-Insensitive Search
- Modified all search patterns to be case-insensitive
- Normalized text before searching (converting to lowercase)
- Added variations of terms with different capitalizations

### Contextual Search
- Implemented window-based search to capture surrounding context
- Added pre and post context to search patterns
- Attempted to identify relevant paragraphs containing target information

### Question Refinement
- Modified question prompts to be more specific
- Added context to questions (e.g., "In the contract, what is the percentage for...")
- Tried different phrasings of the same question
- Added German-specific question formats

Despite these attempts, none of these approaches significantly improved the extraction accuracy. The models still struggled with:
- Complex contractual language
- Information spread across multiple paragraphs
- Implicit information requiring context understanding
- Variations in how the same information is presented

This suggests that the challenge lies not in the search techniques but in the models' fundamental understanding of the contract context and specialized terminology. In plain english, too big of a weight for this small models to lift. I also didn't persue further attempts to improve the accuracy, because I got the strong feeling that I was just overfitting the model to that specific document.

## Current Findings

Our testing with both German language models has revealed:

- The models can successfully identify some basic patterns in the contract text
- They struggle with complex or context-dependent information
- Performance varies significantly depending on how the information is phrased in the contract
- The models often return token markers ([CLS], [SEP]) when they cannot find the requested information
- I should probably prepare my heart for future disappointment when using free "solutions"

## Future Directions

1. **Expand Model Testing**: Test additional German language models, particularly those specialized in legal or technical document analysis (if in the future I found a new one)
2. **Improve Prompt Engineering**: Develop more effective prompts to guide the models toward better extraction (probably the most bang for the buck approach)
3. **Hybrid Approaches**: Investigate combining rule-based extraction with model-based understanding

## Technical Implementation

The project uses:
- Python for implementation
- Hugging Face Transformers library for model access
- PyMuPDF for PDF text extraction
- JSON for structured output

## Usage

```python
from contract_analyzer_gbgq_V3 import ContractAnalyzer

# Initialize the analyzer
analyzer = ContractAnalyzer()

# Process a single contract
result = analyzer.analyze_contract("path/to/contract.pdf")

# Process a directory of contracts
results = analyzer.process_directory("path/to/contracts/")

# Save contract terms to a structured JSON file
contract_terms_path = analyzer.save_contract_terms(results)
```

## Conclusion

This project worked as a sneak-peak the current limitations of small German language models in specialized contract analysis tasks. While these models show promise for basic (really basic) pattern recognition, they struggle with more complex information extraction. The identical performance of the tested models suggests that the challenges may be inherent to the task rather than specific to a particular model architecture.

## Cliffhanger: RAG-Based Solution

Based on the limitations observed in this project, our next iteration will implement a Retrieval-Augmented Generation (RAG) approach.
The RAG approach should address many of the current limitations by:
- Providing better context understanding through semantic search
- Reducing hallucination by grounding responses in retrieved documents
- Handling complex contractual language more effectively
- Supporting more nuanced understanding of contract terms 