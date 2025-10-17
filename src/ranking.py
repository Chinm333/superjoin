"""
Intelligent Result Ranking and Output Formatting

This module handles ranking search results by semantic relevance and formatting
output with meaningful context and explanations. Enhanced with Gemini AI integration.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
try:
    from .semantic_engine import SemanticInfo
    from .query_processor import QueryIntent
    from .gemini_service import GeminiService
    from .config import Config
    from .business_types import BusinessConcept, FormulaType
except ImportError:
    from semantic_engine import SemanticInfo
    from query_processor import QueryIntent
    from gemini_service import GeminiService
    from config import Config
    from business_types import BusinessConcept, FormulaType


class ResultFormat(Enum):
    """Different output formats for search results"""
    STRUCTURED = "structured"  # JSON-like structure
    HUMAN_READABLE = "human_readable"  # Natural language summary
    GROUPED = "grouped"  # Grouped by business concept
    DETAILED = "detailed"  # Full context and explanations


@dataclass
class SearchResult:
    """A single search result with ranking information"""
    semantic_info: SemanticInfo
    relevance_score: float
    ranking_factors: Dict[str, float]
    business_context: str
    location_context: str
    explanation: str


@dataclass
class SearchResults:
    """Collection of search results with metadata"""
    results: List[SearchResult]
    total_found: int
    query_intent: QueryIntent
    search_time: float
    format_type: ResultFormat


class ResultRanker:
    """Ranks and formats search results based on semantic relevance"""
    
    def __init__(self):
        self.ranking_weights = {
            "concept_match": 0.35,
            "formula_type_match": 0.25,
            "confidence_score": 0.20,
            "context_importance": 0.15,
            "location_relevance": 0.05
        }
        
        # Initialize Gemini service for enhanced explanations
        self.gemini_service = GeminiService()
        self.use_gemini = Config.USE_GEMINI and self.gemini_service.is_available
        
        if self.use_gemini:
            print("Enhanced result ranking with Gemini AI")
        else:
            print("Using rule-based result ranking")
    
    def rank_results(self, semantic_results: List[SemanticInfo], 
                    query_intent: QueryIntent) -> List[SearchResult]:
        """Rank semantic results based on query intent"""
        search_results = []
        
        for semantic_info in semantic_results:
            relevance_score, ranking_factors = self._calculate_relevance(semantic_info, query_intent)
            business_context = self._generate_business_context(semantic_info)
            location_context = self._generate_location_context(semantic_info)
            explanation = self._generate_result_explanation(semantic_info, query_intent, ranking_factors)
            
            search_result = SearchResult(
                semantic_info=semantic_info,
                relevance_score=relevance_score,
                ranking_factors=ranking_factors,
                business_context=business_context,
                location_context=location_context,
                explanation=explanation
            )
            
            search_results.append(search_result)
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return search_results
    
    def _calculate_relevance(self, semantic_info: SemanticInfo, 
                           query_intent: QueryIntent) -> Tuple[float, Dict[str, float]]:
        """Calculate relevance score and individual ranking factors"""
        factors = {}
        concept_score = self._calculate_concept_match(semantic_info, query_intent)
        factors["concept_match"] = concept_score
        formula_score = self._calculate_formula_match(semantic_info, query_intent)
        factors["formula_type_match"] = formula_score
        factors["confidence_score"] = semantic_info.confidence_score
        context_score = self._calculate_context_importance(semantic_info)
        factors["context_importance"] = context_score
        
        location_score = self._calculate_location_relevance(semantic_info, query_intent)
        factors["location_relevance"] = location_score
        total_score = sum(
            factors[factor] * weight 
            for factor, weight in self.ranking_weights.items()
        )
        
        return total_score, factors
    
    def _calculate_concept_match(self, semantic_info: SemanticInfo, 
                               query_intent: QueryIntent) -> float:
        """Calculate how well the cell's concepts match the query concepts"""
        if not query_intent.target_concepts:
            return 0.0
        
        if not semantic_info.business_concepts:
            return 0.0
        exact_matches = set(semantic_info.business_concepts) & set(query_intent.target_concepts)
        if exact_matches:
            return 1.0
        
        related_concepts = self._get_related_concepts(query_intent.target_concepts)
        related_matches = set(semantic_info.business_concepts) & set(related_concepts)
        if related_matches:
            return 0.7

        partial_score = 0.0
        for query_concept in query_intent.target_concepts:
            for cell_concept in semantic_info.business_concepts:
                if self._are_concepts_related(query_concept, cell_concept):
                    partial_score = max(partial_score, 0.4)
        
        return partial_score
    
    def _calculate_formula_match(self, semantic_info: SemanticInfo, 
                               query_intent: QueryIntent) -> float:
        """Calculate how well the cell's formula type matches the query"""
        if not query_intent.target_formula_types:
            return 0.0
        
        if not semantic_info.formula_type:
            return 0.0
        
        if semantic_info.formula_type in query_intent.target_formula_types:
            return 1.0
        related_formulas = self._get_related_formula_types(query_intent.target_formula_types)
        if semantic_info.formula_type in related_formulas:
            return 0.6
        
        return 0.0
    
    def _calculate_context_importance(self, semantic_info: SemanticInfo) -> float:
        """Boost context importance for key business formulas (margin, profit, growth, variance, etc.)"""
        score = 0.0
        # Boost for core business concepts from semantic engine:
        key_archetype = False
        if semantic_info.business_concepts:
            for concept in semantic_info.business_concepts:
                if concept in [BusinessConcept.MARGIN, BusinessConcept.PROFIT, BusinessConcept.GROWTH, BusinessConcept.VARIANCE, BusinessConcept.RATIO, BusinessConcept.REVENUE, BusinessConcept.COST]:
                    key_archetype = True
                    break
        if key_archetype and semantic_info.cell_info.is_formula:
            score += 0.8    # High importance to key business formulas
        elif semantic_info.cell_info.is_formula:
            score += 0.4
        elif semantic_info.cell_info.is_header:
            score += 0.2
        score += semantic_info.confidence_score * 0.3
        return min(score, 1.0)
    
    def _calculate_location_relevance(self, semantic_info: SemanticInfo, 
                                    query_intent: QueryIntent) -> float:
        """Calculate relevance based on location and query type"""
        if query_intent.query_type.value == "locational":
            if semantic_info.cell_info.is_header:
                return 0.8
            elif semantic_info.cell_info.row <= 5:  
                return 0.6
            else:
                return 0.3
        
        return 0.5  
    
    def _get_related_concepts(self, concepts: List[BusinessConcept]) -> List[BusinessConcept]:
        """Get concepts related to the target concepts"""
        related = []
        
        concept_relations = {
            BusinessConcept.REVENUE: [BusinessConcept.TOTAL],
            BusinessConcept.PROFIT: [BusinessConcept.MARGIN],
            BusinessConcept.MARGIN: [BusinessConcept.PROFIT, BusinessConcept.PERCENTAGE],
            BusinessConcept.COST: [],
            BusinessConcept.GROWTH: [BusinessConcept.PERCENTAGE],
            BusinessConcept.EFFICIENCY: [BusinessConcept.RATIO]
        }
        
        for concept in concepts:
            if concept in concept_relations:
                related.extend(concept_relations[concept])
        
        return list(set(related))
    
    def _get_related_formula_types(self, formula_types: List[FormulaType]) -> List[FormulaType]:
        """Get formula types related to the target types"""
        related = []
        
        formula_relations = {
            FormulaType.SUM: [FormulaType.CALCULATION],
            FormulaType.AVERAGE: [FormulaType.CALCULATION],
            FormulaType.PERCENTAGE: [FormulaType.RATIO],
            FormulaType.RATIO: [FormulaType.PERCENTAGE],
            FormulaType.GROWTH_RATE: [FormulaType.PERCENTAGE, FormulaType.CALCULATION]
        }
        
        for formula_type in formula_types:
            if formula_type in formula_relations:
                related.extend(formula_relations[formula_type])
        
        return list(set(related))
    
    def _are_concepts_related(self, concept1: BusinessConcept, concept2: BusinessConcept) -> bool:
        """Check if two concepts are related"""
        related_pairs = [
            (BusinessConcept.REVENUE, BusinessConcept.TOTAL),
            (BusinessConcept.PROFIT, BusinessConcept.MARGIN),
            (BusinessConcept.GROWTH, BusinessConcept.PERCENTAGE),
            (BusinessConcept.EFFICIENCY, BusinessConcept.RATIO)
        ]
        
        return (concept1, concept2) in related_pairs or (concept2, concept1) in related_pairs
    
    def _generate_business_context(self, semantic_info: SemanticInfo) -> str:
        """Generate business context for the result"""
        context_parts = []
        
        if semantic_info.business_concepts:
            concept_names = [c.value for c in semantic_info.business_concepts]
            context_parts.append(f"Business concepts: {', '.join(concept_names)}")
        
        if semantic_info.formula_type:
            context_parts.append(f"Formula type: {semantic_info.formula_type.value}")
        
        if semantic_info.cell_info.is_formula and semantic_info.cell_info.formula:
            context_parts.append(f"Formula: {semantic_info.cell_info.formula}")
        
        return "; ".join(context_parts) if context_parts else "No specific business context"
    
    def _generate_location_context(self, semantic_info: SemanticInfo) -> str:
        """Generate location context for the result"""
        cell = semantic_info.cell_info
        location_parts = []
        
        location_parts.append(f"Sheet: {cell.sheet_name}")
        location_parts.append(f"Cell: {cell.cell_address}")
        
        if cell.is_header:
            location_parts.append("(Header cell)")
        
        if cell.row <= 3:
            location_parts.append("(Top section)")
        
        return " | ".join(location_parts)
    
    def _generate_result_explanation(self, semantic_info: SemanticInfo, 
                                   query_intent: QueryIntent, 
                                   ranking_factors: Dict[str, float]) -> str:
        """Generate explanation for why this result matches the query with Gemini AI enhancement"""
        return self._generate_rule_based_explanation(semantic_info, query_intent, ranking_factors)
    
    def _generate_gemini_explanation(self, semantic_info: SemanticInfo, 
                                   query_intent: QueryIntent, 
                                   ranking_factors: Dict[str, float]) -> Optional[str]:
        """Generate explanation using Gemini AI"""
        if hasattr(self.gemini_service, 'quota_exceeded') and self.gemini_service.quota_exceeded:
            return None
            
        try:
            return self.gemini_service.generate_result_explanation(
                semantic_info.cell_info,
                query_intent.original_query,
                semantic_info.business_concepts
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                if hasattr(self.gemini_service, 'quota_exceeded'):
                    self.gemini_service.quota_exceeded = True
                print(f"Warning: Gemini quota exceeded during explanation generation. Using rule-based explanations.")
            else:
                print(f"Gemini explanation generation failed: {e}")
            return None
    
    def _generate_rule_based_explanation(self, semantic_info: SemanticInfo, query_intent: QueryIntent, ranking_factors: Dict[str, float]) -> str:
        """Show business/semantic engine explanation first, then generic match factors."""
        explanation_parts = []
        # Show detailed business meaning from engine
        if getattr(semantic_info, "explanation", None):
            explanation_parts.append(semantic_info.explanation)
        if ranking_factors.get("concept_match", 0) > 0.5:
            explanation_parts.append("Strong concept match")
        elif ranking_factors.get("concept_match", 0) > 0.2:
            explanation_parts.append("Partial concept match")
        if ranking_factors.get("formula_type_match", 0) > 0.5:
            explanation_parts.append("Formula type matches query")
        if ranking_factors.get("confidence_score", 0) > 0.7:
            explanation_parts.append("High confidence semantic analysis")
        if ranking_factors.get("context_importance", 0) > 0.6:
            explanation_parts.append("Important calculation context (business formula block)")
        return "; ".join(str(s) for s in explanation_parts if s) or "General relevance match"


class ResultFormatter:
    """Formats search results for different output types"""
    
    def __init__(self):
        self.formatters = {
            ResultFormat.STRUCTURED: self._format_structured,
            ResultFormat.HUMAN_READABLE: self._format_human_readable,
            ResultFormat.GROUPED: self._format_grouped,
            ResultFormat.DETAILED: self._format_detailed
        }
    
    def format_results(self, search_results: List[SearchResult], 
                      query_intent: QueryIntent, 
                      format_type: ResultFormat = ResultFormat.HUMAN_READABLE) -> str:
        """Format search results according to the specified format"""
        formatter = self.formatters.get(format_type, self._format_human_readable)
        return formatter(search_results, query_intent)
    
    def _format_structured(self, search_results: List[SearchResult], 
                          query_intent: QueryIntent) -> str:
        """Format results as structured JSON-like output"""
        structured_results = {
            "query": query_intent.original_query,
            "query_type": query_intent.query_type.value,
            "total_results": len(search_results),
            "results": []
        }
        
        for result in search_results:
            result_data = {
                "location": {
                    "sheet": result.semantic_info.cell_info.sheet_name,
                    "cell": result.semantic_info.cell_info.cell_address,
                    "row": result.semantic_info.cell_info.row,
                    "col": result.semantic_info.cell_info.col
                },
                "content": {
                    "value": result.semantic_info.cell_info.value,
                    "formula": result.semantic_info.cell_info.formula,
                    "data_type": result.semantic_info.cell_info.data_type
                },
                "semantics": {
                    "business_concepts": [c.value for c in result.semantic_info.business_concepts],
                    "formula_type": result.semantic_info.formula_type.value if result.semantic_info.formula_type else None,
                    "confidence_score": result.semantic_info.confidence_score
                },
                "relevance": {
                    "score": result.relevance_score,
                    "ranking_factors": result.ranking_factors
                },
                "context": {
                    "business_context": result.business_context,
                    "location_context": result.location_context,
                    "explanation": result.explanation
                }
            }
            structured_results["results"].append(result_data)
        
        return json.dumps(structured_results, indent=2, default=str)
    
    def _format_human_readable(self, search_results: List[SearchResult], 
                              query_intent: QueryIntent) -> str:
        """Format results as human-readable text"""
        if not search_results:
            return f"No results found for query: '{query_intent.original_query}'"
        
        output = [f"Search Results for: '{query_intent.original_query}'"]
        output.append(f"Found {len(search_results)} results\n")
        
        for i, result in enumerate(search_results, 1):
            cell = result.semantic_info.cell_info
            
            output.append(f"{i}. {self._get_result_title(result)}")
            output.append(f"   Location: {result.location_context}")
            
            if cell.value is not None:
                output.append(f"   Value: {cell.value}")
            
            if cell.formula:
                output.append(f"   Formula: {cell.formula}")
            
            output.append(f"   Business Context: {result.business_context}")
            output.append(f"   Relevance: {result.relevance_score:.2f} - {result.explanation}")
            output.append("")
        
        return "\n".join(output)
    
    def _format_grouped(self, search_results: List[SearchResult], 
                       query_intent: QueryIntent) -> str:
        """Format results grouped by business concept"""
        if not search_results:
            return f"No results found for query: '{query_intent.original_query}'"
        concept_groups = {}
        for result in search_results:
            concepts = result.semantic_info.business_concepts
            if not concepts:
                concepts = [BusinessConcept.UNKNOWN]
            
            for concept in concepts:
                if concept not in concept_groups:
                    concept_groups[concept] = []
                concept_groups[concept].append(result)
        
        output = [f"Search Results for: '{query_intent.original_query}'"]
        output.append(f"Found {len(search_results)} results grouped by concept\n")
        
        for concept, results in concept_groups.items():
            output.append(f"=== {concept.value.upper()} ===")
            for result in results:
                cell = result.semantic_info.cell_info
                output.append(f"  • {cell.sheet_name}!{cell.cell_address}: {cell.value}")
                if cell.formula:
                    output.append(f"    Formula: {cell.formula}")
            output.append("")
        
        return "\n".join(output)
    
    def _format_detailed(self, search_results: List[SearchResult], 
                        query_intent: QueryIntent) -> str:
        """Format results with full detailed information"""
        if not search_results:
            return f"No results found for query: '{query_intent.original_query}'"
        
        output = [f"Detailed Search Results for: '{query_intent.original_query}'"]
        output.append(f"Query Analysis: {query_intent.explanation}")
        output.append(f"Found {len(search_results)} results\n")
        
        for i, result in enumerate(search_results, 1):
            cell = result.semantic_info.cell_info
            
            output.append(f"{'='*60}")
            output.append(f"RESULT {i} (Relevance: {result.relevance_score:.3f})")
            output.append(f"{'='*60}")
            
            output.append(f"Location: {result.location_context}")
            output.append(f"Content: {cell.value}")
            if cell.formula:
                output.append(f"Formula: {cell.formula}")
            output.append(f"Data Type: {cell.data_type}")
            
            output.append(f"\nSemantic Analysis:")
            output.append(f"  Business Concepts: {[c.value for c in result.semantic_info.business_concepts]}")
            output.append(f"  Formula Type: {result.semantic_info.formula_type.value if result.semantic_info.formula_type else 'None'}")
            output.append(f"  Confidence Score: {result.semantic_info.confidence_score:.3f}")
            
            output.append(f"\nRelevance Analysis:")
            for factor, score in result.ranking_factors.items():
                output.append(f"  {factor}: {score:.3f}")
            
            output.append(f"\nContext:")
            output.append(f"  Business: {result.business_context}")
            output.append(f"  Explanation: {result.explanation}")
            output.append("")
        
        return "\n".join(output)
    
    def _get_result_title(self, result: SearchResult) -> str:
        """Generate a title for a search result"""
        cell = result.semantic_info.cell_info
        concepts = result.semantic_info.business_concepts
        
        if concepts:
            concept_names = [c.value for c in concepts]
            return f"{', '.join(concept_names).title()} ({cell.sheet_name}!{cell.cell_address})"
        elif cell.is_header:
            return f"Header: {cell.value} ({cell.sheet_name}!{cell.cell_address})"
        else:
            return f"Cell {cell.cell_address} ({cell.sheet_name})"


if __name__ == "__main__":
    print("ResultRanker and ResultFormatter initialized successfully!")
    print("Ready to rank and format search results.")
