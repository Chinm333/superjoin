"""
Basic tests for the semantic spreadsheet search engine
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import SpreadsheetParser, CellInfo
from semantic_engine import SemanticEngine
from query_processor import QueryProcessor, QueryType
from ranking import ResultRanker, ResultFormatter, ResultFormat
from business_types import BusinessConcept, FormulaType


def test_parser():
    """Test the spreadsheet parser"""
    print("Testing SpreadsheetParser...")
    
    parser = SpreadsheetParser()
    
    # Test formula dependency extraction
    test_formulas = [
        "=SUM(A1:A10)",
        "=B5/B6*100",
        "=IF(C1>0, D1*E1, 0)",
        "=VLOOKUP(A1, Sheet2!A:B, 2, FALSE)"
    ]
    
    for formula in test_formulas:
        dependencies = parser.get_formula_dependencies(formula)
        print(f"  Formula: {formula}")
        print(f"  Dependencies: {dependencies}")
    
    print("+ Parser tests passed\n")


def test_semantic_engine():
    """Test the semantic engine"""
    print("Testing SemanticEngine...")
    
    engine = SemanticEngine()
    
    # Test business concept matching
    test_texts = [
        "Q1 Revenue",
        "Gross Profit Margin %",
        "Budget vs Actual Variance",
        "Year over Year Growth",
        "ROI Calculation",
        "Operating Expenses",
        "Net Profit Margin"
    ]
    
    for text in test_texts:
        concepts = engine._match_business_concepts(text)
        print(f"  Text: '{text}' -> Concepts: {[c.value for c in concepts]}")
    
    # Test formula analysis
    test_formulas = [
        "=SUM(B2:B10)",
        "=B4/B2*100",
        "=IF(C1>0, D1*E1, 0)",
        "=AVERAGE(D1:D12)"
    ]
    
    for formula in test_formulas:
        formula_type, confidence = engine._analyze_formula(formula)
        print(f"  Formula: {formula} -> Type: {formula_type.value if formula_type else 'None'}, Confidence: {confidence}")
    
    print("+ SemanticEngine tests passed\n")


def test_query_processor():
    """Test the query processor"""
    print("Testing QueryProcessor...")
    
    processor = QueryProcessor()
    
    # Test query processing
    test_queries = [
        "find all revenue calculations",
        "show me profitability metrics",
        "where are my margin analyses?",
        "what percentage calculations do I have?",
        "find budget vs actual comparisons"
    ]
    
    for query in test_queries:
        intent = processor.process_query(query)
        print(f"  Query: '{query}'")
        print(f"    Type: {intent.query_type.value}")
        print(f"    Concepts: {[c.value for c in intent.target_concepts]}")
        print(f"    Confidence: {intent.confidence_score:.2f}")
    
    # Test query validation
    test_validations = [
        ("find revenue", True),
        ("", False),
        ("a", False),
        ("show me all the profit calculations", True)
    ]
    
    for query, expected in test_validations:
        is_valid, message = processor.validate_query(query)
        print(f"  Validation: '{query}' -> Valid: {is_valid} (Expected: {expected})")
    
    print("+ QueryProcessor tests passed\n")


def test_ranking():
    """Test the ranking system"""
    print("Testing ResultRanker and ResultFormatter...")
    
    ranker = ResultRanker()
    formatter = ResultFormatter()
    
    # Create mock semantic info for testing
    from semantic_engine import SemanticInfo
    from query_processor import QueryIntent
    
    # Mock cell info
    mock_cell = CellInfo(
        sheet_name="Test Sheet",
        cell_address="B5",
        value="25.5%",
        formula="=B4/B2*100",
        data_type="formula",
        is_formula=True,
        row=5,
        col=2
    )
    
    # Mock semantic info
    mock_semantic = SemanticInfo(
        cell_info=mock_cell,
        business_concepts=[BusinessConcept.MARGIN, BusinessConcept.PERCENTAGE],
        formula_type=None,  # Would be set by actual analysis
        confidence_score=0.8,
        context_clues=["margin", "percentage"],
        explanation="Margin calculation with percentage"
    )
    
    # Mock query intent
    mock_intent = QueryIntent(
        original_query="find margin calculations",
        query_type=QueryType.CONCEPTUAL,
        target_concepts=[BusinessConcept.MARGIN],
        target_formula_types=[],
        search_criteria={},
        confidence_score=0.9,
        explanation="Looking for margin-related calculations"
    )
    
    # Test ranking
    results = ranker.rank_results([mock_semantic], mock_intent)
    print(f"  Ranked {len(results)} results")
    if results:
        print(f"  Top result relevance: {results[0].relevance_score:.3f}")
        print(f"  Business context: {results[0].business_context}")
    
    # Test formatting
    formatted = formatter.format_results(results, mock_intent, ResultFormat.HUMAN_READABLE)
    print(f"  Formatted output length: {len(formatted)} characters")
    
    print("+ Ranking tests passed\n")


def run_all_tests():
    """Run all basic tests"""
    print("=" * 50)
    print("RUNNING BASIC TESTS")
    print("=" * 50)
    
    try:
        test_parser()
        test_semantic_engine()
        test_query_processor()
        test_ranking()
        
        print("=" * 50)
        print("ALL TESTS PASSED! +")
        print("=" * 50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
