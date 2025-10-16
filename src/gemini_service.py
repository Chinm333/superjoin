"""
Gemini AI Service for enhanced semantic understanding
"""
import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

from config import Config
try:
    from .business_types import BusinessConcept, FormulaType
    from .parser import CellInfo
except ImportError:
    from business_types import BusinessConcept, FormulaType
    from parser import CellInfo


@dataclass
class GeminiAnalysis:
    """Result of Gemini analysis"""
    business_concepts: List[BusinessConcept]
    formula_type: Optional[FormulaType]
    confidence_score: float
    explanation: str
    context_clues: List[str]
    reasoning: str


class GeminiService:
    """Service for interacting with Gemini AI"""
    
    def __init__(self):
        self.client = None
        self.model = None
        self.is_available = False
        self.quota_exceeded = False
        self.request_count = 0
        self.last_request_time = 0
        
        if GEMINI_AVAILABLE and Config.validate_gemini_config():
            self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini client"""
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            self.is_available = True
            print("✅ Gemini AI initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            self.is_available = False
    
    def analyze_cell_semantics(self, cell_info: CellInfo, sheet_context: Dict[str, Any]) -> Optional[GeminiAnalysis]:
        """
        Analyze cell semantics using Gemini AI
        
        Args:
            cell_info: Cell information to analyze
            sheet_context: Context about the sheet (headers, surrounding cells)
            
        Returns:
            GeminiAnalysis object or None if analysis fails
        """
        if not self.is_available or self.quota_exceeded:
            return None
        
        # Check rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < 5:  # 5 second minimum between requests
            time.sleep(5 - (current_time - self.last_request_time))
        
        try:
            prompt = self._create_analysis_prompt(cell_info, sheet_context)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=Config.GEMINI_MAX_TOKENS,
                    temperature=Config.GEMINI_TEMPERATURE
                )
            )
            
            self.request_count += 1
            self.last_request_time = time.time()
            
            return self._parse_gemini_response(response.text, cell_info)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                self.quota_exceeded = True
                print(f"⚠️  Gemini quota exceeded for cell {cell_info.cell_address}. Switching to rule-based analysis for remaining cells.")
                print(f"📊 Total Gemini requests made: {self.request_count}")
            else:
                print(f"Gemini analysis failed for cell {cell_info.cell_address}: {e}")
            return None
    
    def process_natural_language_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Process natural language query using Gemini
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with query analysis or None if processing fails
        """
        if not self.is_available:
            return None
        
        try:
            prompt = self._create_query_analysis_prompt(query)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=Config.GEMINI_MAX_TOKENS,
                    temperature=Config.GEMINI_TEMPERATURE
                )
            )
            
            return self._parse_query_response(response.text)
            
        except Exception as e:
            print(f"Gemini query processing failed: {e}")
            return None
    
    def generate_result_explanation(self, cell_info: CellInfo, query: str, 
                                  business_concepts: List[BusinessConcept]) -> Optional[str]:
        """
        Generate explanation for search result using Gemini
        
        Args:
            cell_info: Cell information
            query: Original search query
            business_concepts: Detected business concepts
            
        Returns:
            Generated explanation or None if generation fails
        """
        if not self.is_available:
            return None
        
        try:
            prompt = self._create_explanation_prompt(cell_info, query, business_concepts)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=Config.GEMINI_TEMPERATURE
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Gemini explanation generation failed: {e}")
            return None
    
    def _create_analysis_prompt(self, cell_info: CellInfo, sheet_context: Dict[str, Any]) -> str:
        """Create prompt for cell semantic analysis"""
        return f"""
Analyze this spreadsheet cell and identify its business meaning. Return ONLY valid JSON.

CELL INFORMATION:
- Address: {cell_info.cell_address}
- Value: {cell_info.value}
- Formula: {cell_info.formula if cell_info.is_formula else 'None'}
- Data Type: {cell_info.data_type}

SHEET CONTEXT:
- Sheet Name: {sheet_context.get('sheet_name', 'Unknown')}
- Headers: {sheet_context.get('headers', [])}
- Row Context: {sheet_context.get('row_context', [])}
- Column Context: {sheet_context.get('column_context', [])}

BUSINESS CONCEPTS TO IDENTIFY:
{', '.join([concept.value for concept in BusinessConcept])}

FORMULA TYPES TO IDENTIFY:
{', '.join([ftype.value for ftype in FormulaType])}

IMPORTANT: Return ONLY valid JSON in this exact format:
{{
    "business_concepts": ["revenue", "cost"],
    "formula_type": "sum",
    "confidence_score": 0.8,
    "explanation": "This cell calculates total revenue",
    "context_clues": ["header mentions revenue", "formula uses SUM"],
    "reasoning": "The cell contains a SUM formula in a revenue column"
}}

Do not include any text before or after the JSON. Focus on business context and meaning.
"""
    
    def _create_query_analysis_prompt(self, query: str) -> str:
        """Create prompt for query analysis"""
        return f"""
Analyze this natural language query for a spreadsheet search engine. Return ONLY valid JSON.

QUERY: "{query}"

BUSINESS CONCEPTS TO MATCH:
{', '.join([concept.value for concept in BusinessConcept])}

QUERY TYPES:
- CONCEPTUAL: Looking for specific business concepts
- FUNCTIONAL: Looking for specific formula types or calculations
- COMPARATIVE: Looking for comparisons or analysis
- LOCATIONAL: Looking for specific locations or time periods

IMPORTANT: Return ONLY valid JSON in this exact format:
{{
    "query_type": "CONCEPTUAL",
    "target_concepts": ["revenue", "profit"],
    "target_formula_types": ["sum", "percentage"],
    "confidence_score": 0.9,
    "search_criteria": {{"time_periods": [], "comparison_types": []}},
    "explanation": "User wants to find revenue and profit calculations"
}}

Do not include any text before or after the JSON. Focus on user intent.
"""
    
    def _create_explanation_prompt(self, cell_info: CellInfo, query: str, 
                                 business_concepts: List[BusinessConcept]) -> str:
        """Create prompt for result explanation"""
        return f"""
Generate a clear, business-focused explanation for why this cell matches the user's query:

USER QUERY: "{query}"

CELL INFORMATION:
- Location: {cell_info.cell_address}
- Value: {cell_info.value}
- Formula: {cell_info.formula if cell_info.is_formula else 'None'}

IDENTIFIED CONCEPTS: {', '.join([concept.value for concept in business_concepts])}

Please provide a concise explanation (2-3 sentences) that:
1. Explains what this cell represents in business terms
2. Connects it to the user's query
3. Provides context about why it's relevant

Make it clear and business-friendly, avoiding technical jargon.
"""
    
    def _parse_gemini_response(self, response_text: str, cell_info: CellInfo) -> Optional[GeminiAnalysis]:
        """Parse Gemini response for cell analysis"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # No JSON found, try to extract information from natural language
                return self._parse_natural_language_response(response_text, cell_info)
            
            json_text = response_text[json_start:json_end]
            
            # Try to fix common JSON issues
            json_text = self._fix_json_format(json_text)
            
            data = json.loads(json_text)
            
            # Parse business concepts
            business_concepts = []
            for concept_name in data.get('business_concepts', []):
                try:
                    concept = BusinessConcept(concept_name.lower())
                    business_concepts.append(concept)
                except ValueError:
                    continue
            
            # Parse formula type
            formula_type = None
            if data.get('formula_type'):
                try:
                    formula_type = FormulaType(data['formula_type'].lower())
                except ValueError:
                    pass
            
            return GeminiAnalysis(
                business_concepts=business_concepts,
                formula_type=formula_type,
                confidence_score=float(data.get('confidence_score', 0.5)),
                explanation=data.get('explanation', ''),
                context_clues=data.get('context_clues', []),
                reasoning=data.get('reasoning', '')
            )
            
        except Exception as e:
            print(f"Failed to parse Gemini response: {e}")
            print(f"Response text: {response_text[:200]}...")
            return None
    
    def _fix_json_format(self, json_text: str) -> str:
        """Fix common JSON formatting issues"""
        # Remove trailing commas
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
        
        # Fix unescaped quotes in strings
        json_text = re.sub(r'([^\\])"([^",:}\]]*)"([^",:}\]]*)"', r'\1"\2\3"', json_text)
        
        return json_text
    
    def _parse_natural_language_response(self, response_text: str, cell_info: CellInfo) -> Optional[GeminiAnalysis]:
        """Parse natural language response when JSON parsing fails"""
        try:
            # Extract business concepts from natural language
            business_concepts = []
            cell_text = str(cell_info.value).lower() if cell_info.value else ""
            
            # Simple keyword matching for fallback
            concept_keywords = {
                'revenue': ['revenue', 'sales', 'income', 'earnings'],
                'cost': ['cost', 'expense', 'spending', 'cogs'],
                'profit': ['profit', 'earnings', 'income'],
                'margin': ['margin', 'percentage'],
                'growth': ['growth', 'increase', 'decrease'],
                'budget': ['budget', 'planned', 'forecast'],
                'actual': ['actual', 'real', 'actuals'],
                'variance': ['variance', 'difference', 'change']
            }
            
            for concept, keywords in concept_keywords.items():
                if any(keyword in cell_text for keyword in keywords):
                    try:
                        business_concepts.append(BusinessConcept(concept))
                    except ValueError:
                        pass

            if not business_concepts:
                response_lower = response_text.lower()
                for concept, keywords in concept_keywords.items():
                    if any(keyword in response_lower for keyword in keywords):
                        try:
                            business_concepts.append(BusinessConcept(concept))
                        except ValueError:
                            pass
            
            return GeminiAnalysis(
                business_concepts=business_concepts,
                formula_type=None,
                confidence_score=0.6,
                explanation=response_text[:200] + "..." if len(response_text) > 200 else response_text,
                context_clues=[],
                reasoning="Parsed from natural language response"
            )
            
        except Exception as e:
            print(f"Failed to parse natural language response: {e}")
            return None
    
    def _parse_query_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse Gemini response for query analysis"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            json_text = response_text[json_start:json_end]
            return json.loads(json_text)
            
        except Exception as e:
            print(f"Failed to parse Gemini query response: {e}")
            return None
    
    def batch_analyze_cells(self, cells: List[CellInfo], sheet_context: Dict[str, Any]) -> List[Optional[GeminiAnalysis]]:
        """
        Analyze multiple cells in batches for efficiency with proper rate limiting for free tier
        
        Args:
            cells: List of cells to analyze
            sheet_context: Sheet context information
            
        Returns:
            List of analysis results (None for failed analyses)
        """
        if not self.is_available:
            return [None] * len(cells)
        
        if not cells:
            return []
        
        results = []
        # Very conservative batch size for free tier (15 requests per minute limit)
        batch_size = 1 
        
        print(f"🔄 Processing {len(cells)} cells with Gemini AI (rate-limited for free tier)")
        
        for i, cell in enumerate(cells):
            if self.quota_exceeded:
                print(f"⚠️  Quota exceeded, skipping remaining {len(cells) - i} cells")
                results.extend([None] * (len(cells) - i))
                break
                
            print(f"   Analyzing cell {i+1}/{len(cells)}: {cell.cell_address}")
            
            analysis = self.analyze_cell_semantics(cell, sheet_context)
            results.append(analysis)
            
            if i < len(cells) - 1: 
                time.sleep(5) 
        
        return results
