import os
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datetime import datetime
import uuid
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import gc
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContractAnalyzer:
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ContractAnalyzer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "deepset/gelectra-base-germanquad"):
        """
        Initialize the ContractAnalyzer with deepset/gelectra-base-germanquad.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        if self._model is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Initialize model and tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self._model.to(self.device)
            
            # Define the questions for each field with more specific prompts
            self.field_questions = {
                "bauwasser": "Wie viel Prozent der Bruttoabrechnungssumme beträgt der Abzug für Bauwasser? Suche nach 'Bauwasser' und dem zugehörigen Prozentsatz.",
                "baustrom": "Wie viel Prozent der Bruttoabrechnungssumme beträgt der Abzug für Baustrom? Suche nach 'Baustrom' und dem zugehörigen Prozentsatz.",
                "bauwesenversicherung": "Wie viel Prozent der Bruttoabrechnungssumme beträgt die Beteiligung an der Bauwesensversicherung? Suche nach 'Bauwesensversicherung' und dem zugehörigen Prozentsatz.",
                "sanitaer": "Wie viel Prozent der Bruttoabrechnungssumme beträgt der Abzug für Sanitär? Suche nach 'Sanitäre Anlagen' und dem zugehörigen Prozentsatz.",
                "bauschild": "Wie hoch ist der Einbehalt für das Firmenschild in Euro? Suche nach 'Firmenschild' und dem zugehörigen Eurobetrag."
            }
            
            # Create output directory if it doesn't exist
            self.output_dir = "output_einbehalt"
            os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise
        finally:
            # Force garbage collection after PDF processing
            gc.collect()
    
    def get_answer(self, context: str, question: str) -> str:
        # Clean and preprocess the context
        cleaned_context = ' '.join(context.split())
        
        # Patterns for different questions
        patterns = {
            'bauwasser': r'(?:Bauwasser|Baustrom).*?(\d+[,.]\d+\s*v\.H\.)',
            'baustrom': r'entspricht\s+(\d+[,.]\d+\s*v\.H\.)\s+für\s+Baustrom',
            'bauwesenversicherung': r'Beteiligung\s+des\s+Auftragnehmers\s+an\s+der\s+Bauwesensversicherung\s+beträgt\s+(\d+[,.]\d+\s*v\.H\.)',
            'sanitaer': r'Sanitäre\s+Anlagen\s+werden\s+bauseits\s+erstellt\s+und\s+unterhalten',
            'bauschild': r'(?:Firmenschild|Bauschild).*?(?:Euro\s+(\d+[,.]\d+(?:\s*[-]\s*|\s+)?(?:€|Euro)?)|(\d+[,.]\d+(?:\s*[-]\s*|\s+)?(?:€|Euro)?))'
        }
        
        # Log the cleaned context for debugging
        logger.info(f"Processing cleaned context for question: {question}")
        
        # Try to find a direct pattern match first
        for key, pattern in patterns.items():
            if key.lower() in question.lower():
                matches = re.finditer(pattern, cleaned_context, re.IGNORECASE)
                for match in matches:
                    if key == 'sanitaer' and match:
                        return "bauseits erstellt"
                    elif match.groups():
                        # Return the first non-None group
                        return next(g for g in match.groups() if g is not None)
        
        # If no pattern match found, use the model as fallback
        try:
            inputs = self._tokenizer.encode_plus(
                question,
                cleaned_context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            outputs = self._model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)
            
            tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer = self._tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error in model processing: {str(e)}")
            return "Information not found"
    
    def analyze_contract(self, pdf_path: str) -> Dict:
        """
        Analyze a single contract and extract relevant information.
        
        Args:
            pdf_path (str): Path to the PDF contract
            
        Returns:
            Dict: Structured data containing extracted information
        """
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            logger.info("Extracted text from PDF:")
            logger.info("=" * 80)
            logger.info(text)
            logger.info("=" * 80)
            
            # Extract fields
            fields = {}
            for field, question in self.field_questions.items():
                answer = self.get_answer(text, question)
                fields[field] = answer
            
            # Create result dictionary
            result = {
                "contract_id": str(uuid.uuid4()),
                "fields": fields,
                "metadata": {
                    "filename": os.path.basename(pdf_path),
                    "processed_date": datetime.now().isoformat(),
                    "model": "deepset/gelectra-base-germanquad"
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing contract {pdf_path}: {str(e)}")
            raise
        finally:
            # Force garbage collection after processing
            gc.collect()
    
    def process_directory(self, directory_path: str, batch_size: int = 5) -> List[Dict]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path (str): Path to the directory containing PDFs
            batch_size (int): Number of PDFs to process before forcing garbage collection
            
        Returns:
            List[Dict]: List of results for each processed contract
        """
        results = []
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        for i, pdf_file in enumerate(tqdm(pdf_files, desc="Processing contracts")):
            pdf_path = os.path.join(directory_path, pdf_file)
            try:
                result = self.analyze_contract(pdf_path)
                results.append(result)
                
                # Force garbage collection after processing batch_size number of files
                if (i + 1) % batch_size == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {str(e)}")
                continue
        
        return results
    
    def save_contract_terms(self, results: List[Dict], output_filename: str = None) -> str:
        """
        Save the contract terms information to a JSON file in a more structured format.
        
        Args:
            results (List[Dict]): List of contract analysis results
            output_filename (str, optional): Custom filename for the output JSON. If None, a timestamp-based name will be used.
            
        Returns:
            str: Path to the saved JSON file
        """
        # Create a structured summary of contract terms
        contract_terms = {
            "summary": {
                "total_contracts": len(results),
                "generated_date": datetime.now().isoformat(),
                "model": self._model.name_or_path
            },
            "contract_terms": []
        }
        
        # Extract and structure the contract terms from each result
        for result in results:
            contract_term = {
                "contract_id": result["contract_id"],
                "filename": result["metadata"]["filename"],
                "processed_date": result["metadata"]["processed_date"],
                "terms": {
                    "bauwasser": {
                        "value": result["fields"].get("bauwasser", "Not found"),
                        "description": "Deduction for construction water as percentage of gross invoice amount"
                    },
                    "baustrom": {
                        "value": result["fields"].get("baustrom", "Not found"),
                        "description": "Deduction for construction electricity as percentage of gross invoice amount"
                    },
                    "bauwesenversicherung": {
                        "value": result["fields"].get("bauwesenversicherung", "Not found"),
                        "description": "Participation in construction insurance as percentage of gross invoice amount"
                    },
                    "sanitaer": {
                        "value": result["fields"].get("sanitaer", "Not found"),
                        "description": "Information about sanitary facilities"
                    },
                    "bauschild": {
                        "value": result["fields"].get("bauschild", "Not found"),
                        "description": "Deduction for company sign in Euro"
                    }
                }
            }
            contract_terms["contract_terms"].append(contract_term)
        
        # Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"contract_terms_{timestamp}.json"
        
        # Save to JSON file
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(contract_terms, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Contract terms saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    logger.info("Starting contract analyzer...")
    analyzer = ContractAnalyzer()
    
    # Process a directory of contracts
    logger.info("Processing directory of contracts...")
    results = analyzer.process_directory("./input_lvv/")
    
    # Save contract terms to a structured JSON file
    logger.info("Saving contract terms to JSON...")
    contract_terms_path = analyzer.save_contract_terms(results)
    logger.info(f"Contract terms saved to: {contract_terms_path}")