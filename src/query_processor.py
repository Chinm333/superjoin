"""
Natural Language Query Processor

This module handles natural language queries and maps them to business concepts
and search criteria for the semantic search engine. Enhanced with Gemini AI integration.
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
try:
    from .semantic_engine import SemanticInfo
    from .gemini_service import GeminiService
    from .config import Config
    from .business_types import BusinessConcept, FormulaType
except ImportError:
    from semantic_engine import SemanticInfo
    from gemini_service import GeminiService
    from config import Config
    from business_types import BusinessConcept, FormulaType


class QueryType(Enum):
    """Types of queries that can be processed"""
    CONCEPTUAL = "conceptual"  # "find profitability metrics"
    FUNCTIONAL = "functional"  # "show percentage calculations"
    COMPARATIVE = "comparative"  # "budget vs actual analysis"
    LOCATIONAL = "locational"  # "where are my margins?"
    TEMPORAL = "temporal"  # "show quarterly data"
    AGGREGATE = "aggregate"  # "find all revenue calculations"


@dataclass
class QueryIntent:
    """Represents the intent behind a user query"""
    original_query: str
    query_type: QueryType
    target_concepts: List[BusinessConcept]
    target_formula_types: List[FormulaType]
    search_criteria: Dict[str, Any]
    confidence_score: float
    explanation: str


class QueryProcessor:
    """Processes natural language queries into searchable criteria"""
    
    def __init__(self):
        self.query_patterns = self._load_query_patterns()
        self.intent_keywords = self._load_intent_keywords()
        self.stop_words = self._load_stop_words()
        
        # Initialize Gemini service
        self.gemini_service = GeminiService()
        self.use_gemini = Config.USE_GEMINI and self.gemini_service.is_available
        
        if self.use_gemini:
            print("Enhanced query processor with Gemini AI")
        else:
            print("Using rule-based query processor")
    
    def _load_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Load patterns for different types of queries"""
        return {
            QueryType.CONCEPTUAL: [
                r"find.*(?:profitability|efficiency|growth|revenue|cost|margin)",
                r"show.*(?:metrics|calculations|analysis)",
                r"what.*(?:profit|margin|ratio|growth)",
                r"where.*(?:revenue|cost|profit|margin)"
            ],
            
            QueryType.FUNCTIONAL: [
                r"show.*(?:percentage|formula|calculation)",
                r"find.*(?:sum|average|count|formula)",
                r"what.*(?:formulas|calculations)",
                r"where.*(?:formula|calculation)"
            ],
            
            QueryType.COMPARATIVE: [
                r"(?:budget|plan|target).*vs.*(?:actual|real)",
                r"(?:compare|comparison).*",
                r"(?:variance|difference|gap)",
                r"(?:benchmark|against)"
            ],
            
            QueryType.LOCATIONAL: [
                r"where.*(?:are|is)",
                r"find.*(?:location|position)",
                r"show.*(?:where|location)"
            ],
            
            QueryType.TEMPORAL: [
                r"(?:quarterly|monthly|yearly|annual)",
                r"(?:q1|q2|q3|q4)",
                r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                r"(?:2023|2024|2025|ytd)"
            ],
            
            QueryType.AGGREGATE: [
                r"all.*(?:calculations|formulas|metrics)",
                r"every.*(?:revenue|cost|profit)",
                r"list.*(?:all|every)",
                r"show.*(?:all|everything)"
            ]
        }
    
    def _load_intent_keywords(self) -> Dict[str, List[str]]:
        """Load keywords that indicate specific intents"""
        return {
            "find": ["find", "locate", "search", "discover", "identify"],
            "show": ["show", "display", "list", "present", "reveal"],
            "what": ["what", "which", "what kind of"],
            "where": ["where", "location", "position"],
            "how": ["how", "how many", "how much"],
            "compare": ["compare", "comparison", "versus", "vs", "against"],
            "analyze": ["analyze", "analysis", "examine", "review"]
        }
    
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words to filter out"""
        return {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these", "those"
        }
    
    def process_query(self, query: str) -> QueryIntent:
        """Process a natural language query into searchable intent with Gemini AI enhancement"""
        
        # For free tier, skip Gemini query processing to avoid quota issues
        # Use rule-based processing which is fast and doesn't consume quota
        return self._process_with_rules(query)
    
    def _process_with_gemini(self, query: str) -> Optional[Dict[str, Any]]:
        """Process query using Gemini AI"""
        # Check if quota is exceeded before making the call
        if hasattr(self.gemini_service, 'quota_exceeded') and self.gemini_service.quota_exceeded:
            return None
            
        try:
            return self.gemini_service.process_natural_language_query(query)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                # Mark quota as exceeded to prevent future calls
                if hasattr(self.gemini_service, 'quota_exceeded'):
                    self.gemini_service.quota_exceeded = True
                print(f"Warning: Gemini quota exceeded during query processing. Using rule-based processing.")
            else:
                print(f"Gemini query processing failed: {e}")
            return None
    
    def _process_with_rules(self, query: str) -> QueryIntent:
        """Fallback rule-based query processing"""
        query_lower = query.lower().strip()
        
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        
        # Extract target concepts
        target_concepts = self._extract_business_concepts(query_lower)
        
        # Extract target formula types
        target_formula_types = self._extract_formula_types(query_lower)
        
        # Extract search criteria
        search_criteria = self._extract_search_criteria(query_lower, query_type)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(query_lower, target_concepts, target_formula_types)
        
        # Generate explanation
        explanation = self._generate_query_explanation(query_type, target_concepts, target_formula_types)
        
        return QueryIntent(
            original_query=query,
            query_type=query_type,
            target_concepts=target_concepts,
            target_formula_types=target_formula_types,
            search_criteria=search_criteria,
            confidence_score=confidence_score,
            explanation=explanation
        )
    
    def _convert_gemini_query_result(self, gemini_result: Dict[str, Any], original_query: str) -> QueryIntent:
        """Convert Gemini query analysis to QueryIntent"""
        try:
            # Parse query type
            query_type_str = gemini_result.get('query_type', 'conceptual').upper()
            query_type = QueryType.CONCEPTUAL  # default
            for qt in QueryType:
                if qt.value.upper() == query_type_str:
                    query_type = qt
                    break
            
            # Parse target concepts
            target_concepts = []
            for concept_name in gemini_result.get('target_concepts', []):
                try:
                    concept = BusinessConcept(concept_name.lower())
                    target_concepts.append(concept)
                except ValueError:
                    continue
            
            # Parse target formula types
            target_formula_types = []
            for formula_type_name in gemini_result.get('target_formula_types', []):
                try:
                    formula_type = FormulaType(formula_type_name.lower())
                    target_formula_types.append(formula_type)
                except ValueError:
                    continue
            
            return QueryIntent(
                original_query=original_query,
                query_type=query_type,
                target_concepts=target_concepts,
                target_formula_types=target_formula_types,
                search_criteria=gemini_result.get('search_criteria', {}),
                confidence_score=float(gemini_result.get('confidence_score', 0.5)),
                explanation=gemini_result.get('explanation', '')
            )
            
        except Exception as e:
            print(f"Failed to convert Gemini query result: {e}")
            # Fallback to rule-based processing
            return self._process_with_rules(original_query)
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1
            scores[query_type] = score
        if scores:
            return max(scores, key=scores.get)
        return QueryType.CONCEPTUAL
    
    def _extract_business_concepts(self, query: str) -> List[BusinessConcept]:
        """Extract business concepts from the query"""
        concepts = []
        
        # Direct concept matching
        concept_keywords = {
            "revenue": BusinessConcept.REVENUE,
            "sales": BusinessConcept.REVENUE,
            "income": BusinessConcept.REVENUE,
            "earnings": BusinessConcept.REVENUE,
            "cost": BusinessConcept.COST,
            "expense": BusinessConcept.COST,
            "spending": BusinessConcept.COST,
            "profit": BusinessConcept.PROFIT,
            "margin": BusinessConcept.MARGIN,
            "ratio": BusinessConcept.RATIO,
            "roi": BusinessConcept.RATIO,
            "growth": BusinessConcept.GROWTH,
            "efficiency": BusinessConcept.EFFICIENCY,
            "productivity": BusinessConcept.EFFICIENCY,
            "budget": BusinessConcept.BUDGET,
            "actual": BusinessConcept.ACTUAL,
            "variance": BusinessConcept.VARIANCE,
            "difference": BusinessConcept.VARIANCE,
            "forecast": BusinessConcept.FORECAST,
            "percentage": BusinessConcept.PERCENTAGE,
            "average": BusinessConcept.AVERAGE,
            "total": BusinessConcept.TOTAL
        }
        
        for keyword, concept in concept_keywords.items():
            if keyword in query:
                concepts.append(concept)
        
        if "profitability" in query:
            concepts.extend([BusinessConcept.PROFIT, BusinessConcept.MARGIN, BusinessConcept.RATIO])
        
        if "efficiency" in query:
            concepts.extend([BusinessConcept.EFFICIENCY, BusinessConcept.RATIO])
        
        if "growth" in query:
            concepts.extend([BusinessConcept.GROWTH, BusinessConcept.PERCENTAGE])
        return list(dict.fromkeys(concepts))
    
    def _extract_formula_types(self, query: str) -> List[FormulaType]:
        """Extract formula types from the query"""
        formula_types = []
        
        formula_keywords = {
            "sum": FormulaType.SUM,
            "total": FormulaType.SUM,
            "average": FormulaType.AVERAGE,
            "mean": FormulaType.AVERAGE,
            "count": FormulaType.COUNT,
            "percentage": FormulaType.PERCENTAGE,
            "percent": FormulaType.PERCENTAGE,
            "ratio": FormulaType.RATIO,
            "growth": FormulaType.GROWTH_RATE,
            "conditional": FormulaType.CONDITIONAL,
            "if": FormulaType.CONDITIONAL,
            "lookup": FormulaType.LOOKUP,
            "vlookup": FormulaType.LOOKUP,
            "calculation": FormulaType.CALCULATION,
            "formula": FormulaType.CALCULATION
        }
        
        for keyword, formula_type in formula_keywords.items():
            if keyword in query:
                formula_types.append(formula_type)
        
        return list(dict.fromkeys(formula_types))
    
    def _extract_search_criteria(self, query: str, query_type: QueryType) -> Dict[str, Any]:
        """Extract specific search criteria from the query"""
        criteria = {}
        
        # Extract time-related criteria
        time_patterns = {
            "quarterly": ["q1", "q2", "q3", "q4", "quarter"],
            "monthly": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            "yearly": ["2023", "2024", "2025", "annual", "yearly"]
        }
        
        for time_type, patterns in time_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    criteria["time_period"] = time_type
                    criteria["specific_period"] = pattern
                    break
        
        if query_type == QueryType.COMPARATIVE:
            if "budget" in query and "actual" in query:
                criteria["comparison_type"] = "budget_vs_actual"
            elif "plan" in query and "actual" in query:
                criteria["comparison_type"] = "plan_vs_actual"
            elif "target" in query and "actual" in query:
                criteria["comparison_type"] = "target_vs_actual"
        
        if query_type == QueryType.LOCATIONAL:
            criteria["include_location"] = True
        
        if query_type == QueryType.AGGREGATE:
            criteria["include_all_matches"] = True
        
        return criteria
    
    def _calculate_confidence(self, query: str, concepts: List[BusinessConcept], 
                            formula_types: List[FormulaType]) -> float:
        """Calculate confidence score for the query interpretation"""
        confidence = 0.0
        
        # Base confidence from having concepts or formula types
        if concepts:
            confidence += 0.4
        if formula_types:
            confidence += 0.3
        
        # Boost confidence for specific business terms
        business_terms = ["revenue", "profit", "margin", "cost", "budget", "actual", "growth"]
        for term in business_terms:
            if term in query:
                confidence += 0.1
        
        # Boost confidence for clear intent words
        intent_words = ["find", "show", "where", "what", "list"]
        for word in intent_words:
            if word in query:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_query_explanation(self, query_type: QueryType, concepts: List[BusinessConcept], 
                                  formula_types: List[FormulaType]) -> str:
        """Generate explanation of what the query is looking for"""
        explanation_parts = []
        
        explanation_parts.append(f"Query type: {query_type.value}")
        
        if concepts:
            concept_names = [c.value for c in concepts]
            explanation_parts.append(f"Looking for: {', '.join(concept_names)}")
        
        if formula_types:
            formula_names = [f.value for f in formula_types]
            explanation_parts.append(f"Formula types: {', '.join(formula_names)}")
        
        return "; ".join(explanation_parts)
    
    def expand_query(self, query_intent: QueryIntent) -> List[str]:
        """Expand a query with synonyms and related terms"""
        expanded_queries = [query_intent.original_query]
        
        # Add variations with synonyms
        synonyms = {
            "find": ["locate", "search", "discover"],
            "show": ["display", "list", "present"],
            "revenue": ["sales", "income", "earnings"],
            "profit": ["earnings", "net income"],
            "cost": ["expense", "spending"],
            "margin": ["profit margin", "gross margin"],
            "growth": ["increase", "change", "yoy"]
        }
        
        original_query = query_intent.original_query.lower()
        
        for original_word, synonym_list in synonyms.items():
            if original_word in original_query:
                for synonym in synonym_list:
                    expanded_query = original_query.replace(original_word, synonym)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
        
        return expanded_queries
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate if a query can be processed"""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query.strip()) < 2:
            return False, "Query too short"
        
        words = query.lower().split()
        meaningful_words = [word for word in words if word not in self.stop_words]
        
        if len(meaningful_words) < 1:
            return False, "Query must contain meaningful words"
        
        return True, "Query is valid"



if __name__ == "__main__":
    processor = QueryProcessor()
    test_queries = [
        "find all revenue calculations",
        "show me profitability metrics",
        "where are my margin analyses?",
        "what percentage calculations do I have?",
        "find budget vs actual comparisons",
        "show efficiency ratios",
        "where is the Q1 revenue data?",
        "find all cost-related formulas"
    ]
    
    print("Testing query processing:")
    for query in test_queries:
        intent = processor.process_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Type: {intent.query_type.value}")
        print(f"Concepts: {[c.value for c in intent.target_concepts]}")
        print(f"Formula Types: {[f.value for f in intent.target_formula_types]}")
        print(f"Confidence: {intent.confidence_score:.2f}")
        print(f"Explanation: {intent.explanation}")
    
    print("\nQueryProcessor initialized successfully!")
    print("Ready to process natural language queries.")
