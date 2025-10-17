"""
Answer Generator for Semantic Spreadsheet Search Engine

This module provides intelligent answer generation by analyzing actual cell values
from search results and providing direct, meaningful answers to user queries.
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import statistics
from decimal import Decimal, InvalidOperation

try:
    from .ranking import SearchResult
    from .query_processor import QueryIntent, QueryType
    from .types import BusinessConcept, FormulaType
    from .gemini_service import GeminiService
    from .config import Config
except ImportError:
    import sys
    import os
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from ranking import SearchResult
    from query_processor import QueryIntent, QueryType
    # Import our custom types module directly
    import importlib.util
    types_spec = importlib.util.spec_from_file_location("custom_types", os.path.join(current_dir, "types.py"))
    custom_types = importlib.util.module_from_spec(types_spec)
    types_spec.loader.exec_module(custom_types)
    BusinessConcept = custom_types.BusinessConcept
    FormulaType = custom_types.FormulaType
    from gemini_service import GeminiService
    from config import Config


class AnswerType(Enum):
    """Types of answers that can be generated"""
    RANKING = "ranking"           # "Which costs are highest?"
    COMPARISON = "comparison"     # "Compare revenue vs costs"
    CALCULATION = "calculation"   # "What's the total profit?"
    SUMMARY = "summary"          # "Show me key metrics"
    TREND = "trend"             # "What are the trends?"
    ANALYSIS = "analysis"        # "Analyze the data"


@dataclass
class CellValue:
    """Represents a cell with its parsed value and metadata"""
    cell_address: str
    sheet_name: str
    raw_value: Any
    parsed_value: Optional[Union[int, float, str]]
    is_numeric: bool
    label: str
    business_concept: Optional[BusinessConcept]
    formula: Optional[str]
    
    def __post_init__(self):
        if self.parsed_value is None:
            self.parsed_value = self._parse_value()
            self.is_numeric = isinstance(self.parsed_value, (int, float))
    
    def _parse_value(self) -> Optional[Union[int, float, str]]:
        """Parse the raw value into a usable format"""
        if self.raw_value is None:
            return None
        
        # Try to parse as number
        try:
            # Handle currency symbols and formatting
            if isinstance(self.raw_value, str):
                # Remove currency symbols, commas, and other formatting
                cleaned = re.sub(r'[$,%()]', '', self.raw_value.strip())
                if cleaned:
                    return float(cleaned)
            elif isinstance(self.raw_value, (int, float)):
                return float(self.raw_value)
        except (ValueError, TypeError):
            pass
        
        # Return as string if not numeric
        return str(self.raw_value) if self.raw_value is not None else None


@dataclass
class AnswerData:
    """Structured data for generating answers"""
    answer_type: AnswerType
    query_intent: QueryIntent
    cell_values: List[CellValue]
    analysis_results: Dict[str, Any]
    confidence: float


class AnswerGenerator:
    """Generates direct answers by analyzing cell values from search results"""
    
    def __init__(self):
        # Initialize Gemini service for enhanced answer generation
        self.gemini_service = GeminiService()
        self.use_gemini = Config.USE_GEMINI and self.gemini_service.is_available
        
        if self.use_gemini:
            print("Enhanced answer generator with Gemini AI")
        else:
            print("Using rule-based answer generation")
    
    def generate_answer(self, search_results: List[SearchResult], 
                       query_intent: QueryIntent) -> str:
        """
        Generate a direct answer from search results
        
        Args:
            search_results: List of search results from the semantic engine
            query_intent: The processed query intent
            
        Returns:
            Natural language answer string
        """
        if not search_results:
            return "I couldn't find any relevant data to answer your question."
        
        # Extract and parse cell values
        cell_values = self._extract_cell_values(search_results)
        
        if not cell_values:
            return "I found relevant cells but couldn't extract meaningful values to answer your question."
        
        # Determine answer type based on query
        answer_type = self._determine_answer_type(query_intent, cell_values)
        
        # Analyze the data
        analysis_results = self._analyze_data(cell_values, answer_type, query_intent)
        
        # Generate the answer
        if self.use_gemini:
            answer = self._generate_with_gemini(cell_values, analysis_results, query_intent, answer_type)
        else:
            answer = self._generate_with_rules(cell_values, analysis_results, query_intent, answer_type)
        
        return answer
    
    def _extract_cell_values(self, search_results: List[SearchResult]) -> List[CellValue]:
        """Extract and parse values from search results"""
        cell_values = []
        
        for result in search_results:
            cell_info = result.semantic_info.cell_info
            
            # Skip if no value
            if cell_info.value is None:
                continue
            
            # Extract label from business context or cell address
            label = self._extract_label(result)
            
            # Determine business concept
            business_concept = None
            if result.semantic_info.business_concepts:
                business_concept = result.semantic_info.business_concepts[0]
            
            cell_value = CellValue(
                cell_address=cell_info.cell_address,
                sheet_name=cell_info.sheet_name,
                raw_value=cell_info.value,
                parsed_value=None,  # Will be parsed in __post_init__
                is_numeric=False,   # Will be set in __post_init__
                label=label,
                business_concept=business_concept,
                formula=cell_info.formula
            )
            
            cell_values.append(cell_value)
        
        return cell_values
    
    def _extract_label(self, result: SearchResult) -> str:
        """Extract a meaningful label for the cell"""
        # Try to get label from business context
        business_context = result.business_context
        if business_context and "Business concepts:" in business_context:
            # Extract the first business concept as label
            concepts_part = business_context.split("Business concepts:")[1].split(";")[0].strip()
            if concepts_part:
                return concepts_part.split(",")[0].strip()
        
        # Try to get from location context
        location_context = result.location_context
        if "Sheet:" in location_context:
            sheet_part = location_context.split("Sheet:")[1].split("|")[0].strip()
            return sheet_part
        
        # Fallback to cell address
        return result.semantic_info.cell_info.cell_address
    
    def _determine_answer_type(self, query_intent: QueryIntent, 
                              cell_values: List[CellValue]) -> AnswerType:
        """Determine what type of answer to generate"""
        query_lower = query_intent.original_query.lower()
        
        # Ranking queries
        if any(word in query_lower for word in ["highest", "lowest", "top", "bottom", "largest", "smallest", "biggest"]):
            return AnswerType.RANKING
        
        # Comparison queries
        if any(word in query_lower for word in ["compare", "vs", "versus", "against", "difference"]):
            return AnswerType.COMPARISON
        
        # Calculation queries
        if any(word in query_lower for word in ["total", "sum", "calculate", "what is", "how much"]):
            return AnswerType.CALCULATION
        
        # Trend queries
        if any(word in query_lower for word in ["trend", "over time", "growth", "change"]):
            return AnswerType.TREND
        
        # Analysis queries
        if any(word in query_lower for word in ["analyze", "analysis", "insights", "findings"]):
            return AnswerType.ANALYSIS
        
        # Default to summary
        return AnswerType.SUMMARY
    
    def _analyze_data(self, cell_values: List[CellValue], 
                     answer_type: AnswerType, 
                     query_intent: QueryIntent) -> Dict[str, Any]:
        """Analyze the cell values based on answer type"""
        analysis = {
            "total_cells": len(cell_values),
            "numeric_cells": len([cv for cv in cell_values if cv.is_numeric]),
            "text_cells": len([cv for cv in cell_values if not cv.is_numeric])
        }
        
        numeric_values = [cv.parsed_value for cv in cell_values if cv.is_numeric and cv.parsed_value is not None]
        
        if numeric_values:
            analysis.update({
                "min_value": min(numeric_values),
                "max_value": max(numeric_values),
                "sum": sum(numeric_values),
                "average": statistics.mean(numeric_values),
                "median": statistics.median(numeric_values),
                "count": len(numeric_values)
            })
        
        # Specific analysis based on answer type
        if answer_type == AnswerType.RANKING:
            analysis["ranked_values"] = self._rank_values(cell_values)
        elif answer_type == AnswerType.COMPARISON:
            analysis["comparisons"] = self._compare_values(cell_values)
        elif answer_type == AnswerType.CALCULATION:
            analysis["calculations"] = self._calculate_metrics(cell_values)
        
        return analysis
    
    def _rank_values(self, cell_values: List[CellValue]) -> List[Tuple[str, float, str]]:
        """Rank numeric values from highest to lowest"""
        numeric_cells = [(cv.label, cv.parsed_value, cv.cell_address) 
                        for cv in cell_values 
                        if cv.is_numeric and cv.parsed_value is not None]
        
        # Sort by value (descending)
        return sorted(numeric_cells, key=lambda x: x[1], reverse=True)
    
    def _compare_values(self, cell_values: List[CellValue]) -> Dict[str, Any]:
        """Compare values and find differences"""
        numeric_cells = [cv for cv in cell_values if cv.is_numeric and cv.parsed_value is not None]
        
        if len(numeric_cells) < 2:
            return {"error": "Need at least 2 numeric values to compare"}
        
        values = [cv.parsed_value for cv in numeric_cells]
        labels = [cv.label for cv in numeric_cells]
        
        return {
            "highest": (labels[values.index(max(values))], max(values)),
            "lowest": (labels[values.index(min(values))], min(values)),
            "difference": max(values) - min(values),
            "ratio": max(values) / min(values) if min(values) != 0 else float('inf')
        }
    
    def _calculate_metrics(self, cell_values: List[CellValue]) -> Dict[str, Any]:
        """Calculate various metrics from the data"""
        numeric_values = [cv.parsed_value for cv in cell_values 
                         if cv.is_numeric and cv.parsed_value is not None]
        
        if not numeric_values:
            return {"error": "No numeric values found for calculations"}
        
        return {
            "total": sum(numeric_values),
            "average": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "range": max(numeric_values) - min(numeric_values)
        }
    
    def _generate_with_rules(self, cell_values: List[CellValue], 
                           analysis_results: Dict[str, Any],
                           query_intent: QueryIntent,
                           answer_type: AnswerType) -> str:
        """Generate answer using rule-based approach"""
        if answer_type == AnswerType.RANKING:
            return self._generate_ranking_answer(cell_values, analysis_results)
        elif answer_type == AnswerType.COMPARISON:
            return self._generate_comparison_answer(cell_values, analysis_results)
        elif answer_type == AnswerType.CALCULATION:
            return self._generate_calculation_answer(cell_values, analysis_results)
        elif answer_type == AnswerType.SUMMARY:
            return self._generate_summary_answer(cell_values, analysis_results)
        else:
            return self._generate_general_answer(cell_values, analysis_results)
    
    def _generate_ranking_answer(self, cell_values: List[CellValue], 
                               analysis_results: Dict[str, Any]) -> str:
        """Generate answer for ranking queries"""
        if "ranked_values" not in analysis_results or not analysis_results["ranked_values"]:
            return "I couldn't find numeric values to rank."
        
        ranked = analysis_results["ranked_values"]
        answer = "Answer: Based on your data, here are the rankings:\n\n"
        
        for i, (label, value, cell_address) in enumerate(ranked[:5], 1):  # Top 5
            answer += f"{i}. **{label}**: {value:,.2f}\n"
        
        if len(ranked) > 5:
            answer += f"\n*Showing top 5 of {len(ranked)} total values*"
        
        return answer
    
    def _generate_comparison_answer(self, cell_values: List[CellValue], 
                                  analysis_results: Dict[str, Any]) -> str:
        """Generate answer for comparison queries"""
        if "comparisons" not in analysis_results:
            return "I couldn't perform the comparison you requested."
        
        comp = analysis_results["comparisons"]
        if "error" in comp:
            return f"Comparison error: {comp['error']}"
        
        answer = "Answer: Here's the comparison:\n\n"
        answer += f"• **Highest**: {comp['highest'][0]} = {comp['highest'][1]:,.2f}\n"
        answer += f"• **Lowest**: {comp['lowest'][0]} = {comp['lowest'][1]:,.2f}\n"
        answer += f"• **Difference**: {comp['difference']:,.2f}\n"
        
        if comp['ratio'] != float('inf'):
            answer += f"• **Ratio**: {comp['highest'][0]} is {comp['ratio']:.1f}x larger than {comp['lowest'][0]}\n"
        
        return answer
    
    def _generate_calculation_answer(self, cell_values: List[CellValue], 
                                   analysis_results: Dict[str, Any]) -> str:
        """Generate answer for calculation queries"""
        if "calculations" not in analysis_results:
            return "I couldn't perform the calculation you requested."
        
        calc = analysis_results["calculations"]
        if "error" in calc:
            return f"Calculation error: {calc['error']}"
        
        answer = "Answer: Here are the calculations:\n\n"
        answer += f"• **Total**: {calc['total']:,.2f}\n"
        answer += f"• **Average**: {calc['average']:,.2f}\n"
        answer += f"• **Median**: {calc['median']:,.2f}\n"
        answer += f"• **Range**: {calc['min']:,.2f} to {calc['max']:,.2f}\n"
        
        return answer
    
    def _generate_summary_answer(self, cell_values: List[CellValue], 
                               analysis_results: Dict[str, Any]) -> str:
        """Generate answer for summary queries"""
        answer = "Answer: Here's a summary of your data:\n\n"
        
        answer += f"• **Total cells found**: {analysis_results['total_cells']}\n"
        answer += f"• **Numeric values**: {analysis_results['numeric_cells']}\n"
        answer += f"• **Text values**: {analysis_results['text_cells']}\n"
        
        if analysis_results['numeric_cells'] > 0:
            answer += f"• **Sum**: {analysis_results['sum']:,.2f}\n"
            answer += f"• **Average**: {analysis_results['average']:,.2f}\n"
            answer += f"• **Range**: {analysis_results['min_value']:,.2f} to {analysis_results['max_value']:,.2f}\n"
        
        return answer
    
    def _generate_general_answer(self, cell_values: List[CellValue], 
                               analysis_results: Dict[str, Any]) -> str:
        """Generate a general answer when specific type isn't determined"""
        answer = "Answer: I found the following data:\n\n"
        
        # Show top values
        numeric_cells = [cv for cv in cell_values if cv.is_numeric and cv.parsed_value is not None]
        if numeric_cells:
            sorted_cells = sorted(numeric_cells, key=lambda x: x.parsed_value, reverse=True)
            answer += "**Key Values:**\n"
            for cv in sorted_cells[:3]:
                answer += f"• {cv.label}: {cv.parsed_value:,.2f}\n"
        
        # Show text values
        text_cells = [cv for cv in cell_values if not cv.is_numeric]
        if text_cells:
            answer += "\n**Categories Found:**\n"
            for cv in text_cells[:5]:
                answer += f"• {cv.label}: {cv.parsed_value}\n"
        
        return answer
    
    def _generate_with_gemini(self, cell_values: List[CellValue], 
                            analysis_results: Dict[str, Any],
                            query_intent: QueryIntent,
                            answer_type: AnswerType) -> Optional[str]:
        """Generate answer using Gemini AI"""
        if not self.gemini_service.is_available:
            return self._generate_with_rules(cell_values, analysis_results, query_intent, answer_type)
        
        try:
            prompt = self._create_answer_prompt(cell_values, analysis_results, query_intent, answer_type)
            response = self.gemini_service.model.generate_content(
                prompt,
                generation_config=self.gemini_service.model._generation_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Gemini answer generation failed: {e}")
            return self._generate_with_rules(cell_values, analysis_results, query_intent, answer_type)
    
    def _create_answer_prompt(self, cell_values: List[CellValue], 
                            analysis_results: Dict[str, Any],
                            query_intent: QueryIntent,
                            answer_type: AnswerType) -> str:
        """Create prompt for Gemini to generate answers"""
        
        # Prepare data summary
        numeric_data = []
        text_data = []
        
        for cv in cell_values:
            if cv.is_numeric and cv.parsed_value is not None:
                numeric_data.append(f"{cv.label}: {cv.parsed_value}")
            else:
                text_data.append(f"{cv.label}: {cv.parsed_value}")
        
        prompt = f"""
You are an expert financial analyst. Generate a clear, direct answer to the user's question based on the spreadsheet data provided.

USER QUESTION: "{query_intent.original_query}"

DATA FOUND:
Numeric Values:
{chr(10).join(numeric_data[:10])}

Text Categories:
{chr(10).join(text_data[:10])}

ANALYSIS RESULTS:
{analysis_results}

ANSWER TYPE: {answer_type.value}

INSTRUCTIONS:
1. Provide a direct, specific answer to the user's question
2. Use the actual values from the data
3. Be concise but informative
4. Format numbers with commas for readability
5. Start with "Answer:" 
6. Use bullet points for multiple items
7. If ranking is requested, show the top items clearly
8. If comparison is requested, highlight the key differences

Generate a professional, helpful answer:
"""
        
        return prompt
