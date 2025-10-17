"""
Semantic Understanding Engine

This module provides business concept recognition, formula semantics analysis,
and context interpretation for spreadsheet content. Enhanced with Gemini AI integration.
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
try:
    from .parser import CellInfo, SheetInfo
    from .gemini_service import GeminiService, GeminiAnalysis
    from .config import Config
    from .business_types import BusinessConcept, FormulaType
except ImportError:
    from parser import CellInfo, SheetInfo
    from gemini_service import GeminiService, GeminiAnalysis
    from config import Config
    from business_types import BusinessConcept, FormulaType


@dataclass
class SemanticInfo:
    """Semantic information about a cell or range"""
    cell_info: CellInfo
    business_concepts: List[BusinessConcept]
    formula_type: Optional[FormulaType] = None
    confidence_score: float = 0.0
    context_clues: List[str] = None
    explanation: str = ""


class SemanticEngine:
    """Engine for understanding business semantics in spreadsheets"""
    
    def __init__(self):
        self.business_vocabulary = self._load_business_vocabulary()
        self.formula_patterns = self._load_formula_patterns()
        self.context_patterns = self._load_context_patterns()
        
        # Initialize Gemini service
        self.gemini_service = GeminiService()
        self.use_gemini = Config.USE_GEMINI and self.gemini_service.is_available
        
        if self.use_gemini:
            print("Enhanced semantic engine with Gemini AI")
        else:
            print("Using rule-based semantic engine")
    
    def _load_business_vocabulary(self) -> Dict[BusinessConcept, Dict[str, Any]]:
        """Load business vocabulary and concept mappings"""
        return {
            BusinessConcept.REVENUE: {
                "keywords": [
                    "revenue", "sales", "income", "earnings", "turnover", "gross sales",
                    "net sales", "total sales", "sales revenue", "top line"
                ],
                "synonyms": {
                    "sales": "revenue",
                    "income": "revenue", 
                    "earnings": "revenue",
                    "turnover": "revenue"
                },
                "formula_indicators": ["SUM", "total", "sum of"],
                "context_clues": ["revenue", "sales", "income", "earnings"]
            },
            
            BusinessConcept.COST: {
                "keywords": [
                    "cost", "expense", "spending", "outlay", "expenditure", "cogs",
                    "cost of goods sold", "operating expense", "overhead", "fixed cost",
                    "variable cost", "direct cost", "indirect cost"
                ],
                "synonyms": {
                    "expense": "cost",
                    "spending": "cost",
                    "outlay": "cost",
                    "expenditure": "cost"
                },
                "formula_indicators": ["SUM", "total", "sum of"],
                "context_clues": ["cost", "expense", "spending", "cogs"]
            },
            
            BusinessConcept.PROFIT: {
                "keywords": [
                    "profit", "earnings", "net income", "net profit", "gross profit",
                    "operating profit", "ebitda", "ebit", "bottom line"
                ],
                "synonyms": {
                    "earnings": "profit",
                    "net income": "profit",
                    "net profit": "profit"
                },
                "formula_indicators": ["-", "subtract", "minus"],
                "context_clues": ["profit", "earnings", "net", "gross"]
            },
            
            BusinessConcept.MARGIN: {
                "keywords": [
                    "margin", "profit margin", "gross margin", "net margin", "operating margin",
                    "contribution margin", "margin %", "margin percentage"
                ],
                "synonyms": {
                    "profit margin": "margin",
                    "gross margin": "margin",
                    "net margin": "margin"
                },
                "formula_indicators": ["/", "divide", "percentage"],
                "context_clues": ["margin", "%", "percentage"]
            },
            
            BusinessConcept.RATIO: {
                "keywords": [
                    "ratio", "roi", "roe", "roa", "debt to equity", "current ratio",
                    "quick ratio", "asset turnover", "inventory turnover"
                ],
                "synonyms": {
                    "roi": "ratio",
                    "roe": "ratio",
                    "roa": "ratio"
                },
                "formula_indicators": ["/", "divide", "ratio"],
                "context_clues": ["ratio", "roi", "roe", "roa"]
            },
            
            BusinessConcept.GROWTH: {
                "keywords": [
                    "growth", "growth rate", "yoy", "year over year", "qoq", "quarter over quarter",
                    "cagr", "compound annual growth", "increase", "decrease", "change"
                ],
                "synonyms": {
                    "yoy": "growth",
                    "year over year": "growth",
                    "qoq": "growth",
                    "quarter over quarter": "growth"
                },
                "formula_indicators": ["/", "-", "percentage", "change"],
                "context_clues": ["growth", "yoy", "qoq", "cagr", "change"]
            },
            
            BusinessConcept.EFFICIENCY: {
                "keywords": [
                    "efficiency", "productivity", "utilization", "performance", "roi",
                    "return on investment", "return on equity", "asset efficiency"
                ],
                "synonyms": {
                    "productivity": "efficiency",
                    "utilization": "efficiency",
                    "performance": "efficiency"
                },
                "formula_indicators": ["/", "ratio", "efficiency"],
                "context_clues": ["efficiency", "productivity", "roi", "return"]
            },
            
            BusinessConcept.BUDGET: {
                "keywords": [
                    "budget", "planned", "target", "forecast", "projection", "estimate"
                ],
                "synonyms": {
                    "planned": "budget",
                    "target": "budget",
                    "forecast": "budget"
                },
                "formula_indicators": ["SUM", "total"],
                "context_clues": ["budget", "planned", "target", "forecast"]
            },
            
            BusinessConcept.ACTUAL: {
                "keywords": [
                    "actual", "realized", "achieved", "real", "current", "year to date", "ytd"
                ],
                "synonyms": {
                    "realized": "actual",
                    "achieved": "actual",
                    "real": "actual"
                },
                "formula_indicators": ["SUM", "total"],
                "context_clues": ["actual", "realized", "achieved", "ytd"]
            },
            
            BusinessConcept.VARIANCE: {
                "keywords": [
                    "variance", "difference", "gap", "deviation", "vs", "versus", "budget vs actual"
                ],
                "synonyms": {
                    "difference": "variance",
                    "gap": "variance",
                    "deviation": "variance"
                },
                "formula_indicators": ["-", "subtract", "difference"],
                "context_clues": ["variance", "difference", "vs", "versus"]
            }
        }
    
    def _load_formula_patterns(self) -> Dict[FormulaType, Dict[str, Any]]:
        """Load patterns for recognizing different types of formulas"""
        return {
            FormulaType.SUM: {
                "patterns": [r"SUM\(", r"SUMIF\(", r"SUMIFS\("],
                "description": "Summation formulas",
                "business_meaning": "Total calculations"
            },
            
            FormulaType.AVERAGE: {
                "patterns": [r"AVERAGE\(", r"AVERAGEIF\(", r"AVERAGEIFS\("],
                "description": "Average calculations",
                "business_meaning": "Central tendency metrics"
            },
            
            FormulaType.COUNT: {
                "patterns": [r"COUNT\(", r"COUNTIF\(", r"COUNTIFS\("],
                "description": "Counting formulas",
                "business_meaning": "Quantity metrics"
            },
            
            FormulaType.PERCENTAGE: {
                "patterns": [r"/.*\*100", r"PERCENTAGE\(", r".*%.*"],
                "description": "Percentage calculations",
                "business_meaning": "Proportional metrics"
            },
            
            FormulaType.RATIO: {
                "patterns": [r".*\/.*", r"RATIO\("],
                "description": "Ratio calculations",
                "business_meaning": "Comparative metrics"
            },
            
            FormulaType.GROWTH_RATE: {
                "patterns": [r"\(.*-.*\)\/.*", r"GROWTH\("],
                "description": "Growth rate calculations",
                "business_meaning": "Change over time metrics"
            },
            
            FormulaType.CONDITIONAL: {
                "patterns": [r"IF\(", r"IFS\(", r"SUMIF\(", r"COUNTIF\("],
                "description": "Conditional formulas",
                "business_meaning": "Conditional logic"
            },
            
            FormulaType.LOOKUP: {
                "patterns": [r"VLOOKUP\(", r"HLOOKUP\(", r"INDEX\(", r"MATCH\("],
                "description": "Lookup formulas",
                "business_meaning": "Data retrieval"
            }
        }
    
    def _load_context_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for understanding context"""
        return {
            "time_periods": [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec",
                "q1", "q2", "q3", "q4", "quarter", "monthly", "yearly",
                "2023", "2024", "2025", "ytd", "year to date"
            ],
            "units": [
                "%", "percent", "percentage", "$", "dollar", "usd", "eur", "gbp",
                "million", "billion", "thousand", "k", "m", "b"
            ],
            "comparisons": [
                "vs", "versus", "against", "compared to", "budget vs actual",
                "plan vs actual", "target vs actual"
            ]
        }
    
    def analyze_cell_semantics(self, cell_info: CellInfo, sheet_context: SheetInfo) -> SemanticInfo:
        """Analyze the semantic meaning of a single cell with Gemini AI enhancement"""
        
        # Use Gemini only for important cells to avoid quota limits
        if self.use_gemini and self._should_use_gemini(cell_info):
            gemini_analysis = self._analyze_with_gemini(cell_info, sheet_context)
            if gemini_analysis:
                return self._convert_gemini_analysis(gemini_analysis, cell_info)
        
        return self._analyze_with_rules(cell_info, sheet_context)
    
    def _should_use_gemini(self, cell_info: CellInfo) -> bool:
        """Determine if Gemini should be used for this cell (very selective for free tier)"""
        if cell_info.row == 1 and cell_info.value:
            cell_text = str(cell_info.value).lower()
            critical_header_keywords = ['profit', 'margin', 'roi', 'ebitda', 'revenue', 'growth', 'budget', 'actual']
            if any(keyword in cell_text for keyword in critical_header_keywords):
                return True
        if cell_info.is_formula and cell_info.formula:
            formula = cell_info.formula.upper()
            if any(complex_func in formula for complex_func in ['IF(', 'VLOOKUP(', 'INDEX(', 'MATCH(']):
                return True
        
        return False
    
    def _analyze_with_gemini(self, cell_info: CellInfo, sheet_context: SheetInfo) -> Optional[GeminiAnalysis]:
        """Analyze cell using Gemini AI"""
        try:
            sheet_context_dict = {
                'sheet_name': getattr(sheet_context, 'name', 'Unknown'),
                'headers': self._get_headers_safe(sheet_context),
                'row_context': self._get_row_context(cell_info, sheet_context),
                'column_context': self._get_column_context(cell_info, sheet_context)
            }
            
            return self.gemini_service.analyze_cell_semantics(cell_info, sheet_context_dict)
            
        except Exception as e:
            print(f"Gemini analysis failed for cell {cell_info.cell_address}: {e}")
            return None
    
    def _get_headers_safe(self, sheet_context: SheetInfo) -> List[str]:
        """Safely extract headers from sheet context"""
        try:
            if hasattr(sheet_context, 'headers') and sheet_context.headers:
                headers = []
                for header in sheet_context.headers:
                    if hasattr(header, 'value'):
                        headers.append(str(header.value))
                    else:
                        headers.append(str(header))
                return headers
            return []
        except:
            return []
    
    def _analyze_with_rules(self, cell_info: CellInfo, sheet_context: SheetInfo) -> SemanticInfo:
        """Enhanced rule-based analysis with header/context-aware semantics"""
        business_concepts = []
        formula_type = None
        confidence_score = 0.0
        context_clues = []
        explanation = ""

        # --- Rich formula semantics (NEW) ---
        resolved_semantics = self._rich_formula_semantics(cell_info, sheet_context)

        if resolved_semantics:
            business_concepts.extend(resolved_semantics.get("business_concepts", []))
            formula_type = resolved_semantics.get("formula_type")
            confidence_score += resolved_semantics.get("confidence", 0)
            if "explanation" in resolved_semantics:
                explanation = resolved_semantics["explanation"]
            if "context_clues" in resolved_semantics:
                context_clues += resolved_semantics["context_clues"]
        else:
            # fallback to previous logic
            if cell_info.is_formula and cell_info.formula:
                formula_type, formula_confidence = self._analyze_formula(cell_info.formula)
                confidence_score += formula_confidence * 0.4
            cell_text = str(cell_info.value).lower() if cell_info.value else ""
            concept_matches = self._match_business_concepts(cell_text)
            business_concepts.extend(concept_matches)
            context_analysis = self._analyze_context(cell_info, sheet_context)
            context_clues.extend(context_analysis["clues"])
            confidence_score += context_analysis["confidence"] * 0.3
            header_analysis = self._analyze_headers(cell_info, sheet_context)
            business_concepts.extend(header_analysis["concepts"])
            confidence_score += header_analysis["confidence"] * 0.3
            business_concepts = list(dict.fromkeys(business_concepts))
            explanation = self._generate_explanation(cell_info, business_concepts, formula_type, context_clues)

        return SemanticInfo(
            cell_info=cell_info,
            business_concepts=business_concepts,
            formula_type=formula_type,
            confidence_score=min(confidence_score, 1.0),
            context_clues=context_clues,
            explanation=explanation
        )

    def _rich_formula_semantics(self, cell_info: CellInfo, sheet_context: SheetInfo) -> Optional[dict]:
        """Advanced, header-aware semantic decoding of formulas"""
        # Only run if we have header context and this is a formula cell
        if not getattr(cell_info, 'is_formula', False) or not getattr(cell_info, 'formula', None):
            return None
        formula = cell_info.formula
        formula_str = str(formula).upper()
        source_refs = []
        try:
            source_refs = sheet_context.parser.get_formula_dependencies(formula) if hasattr(sheet_context, 'parser') else []
        except Exception:
            pass
        # Find referenced cells, use their headers if available
        src_headers = []
        for ref in source_refs:
            ref_cell = None
            for c in sheet_context.cells:
                if c.cell_address.upper() == ref.split('!')[-1].upper():
                    ref_cell = c
                    break
            if ref_cell:
                if ref_cell.row_header:
                    src_headers.append(ref_cell.row_header)
                if ref_cell.column_header:
                    src_headers.append(ref_cell.column_header)
        # Heuristics: arithmetic ratios, differences = margin, growth, variance
        result = {}
        clues = []
        expl = []
        matched = False
        # Margin: (profit/revenue), (gross/net margin)
        if ('/' in formula_str or '*100' in formula_str) and any(s for s in src_headers if s and 'profit' in s.lower()) and any(s for s in src_headers if s and 'revenue' in s.lower()):
            result['business_concepts'] = [BusinessConcept.MARGIN, BusinessConcept.PROFIT]
            result['formula_type'] = FormulaType.PERCENTAGE
            result['confidence'] = 0.9
            clues.append("Header context = margin/profit calculation as profit divided by revenue")
            expl.append("Calculates margin as profit divided by revenue (header-aware)")
            matched = True
        # Growth: (current-prev)/prev
        elif re.search(r"=\(([^-]+)-([^)]+)\)/([^)]+)", formula_str):
            result['business_concepts'] = [BusinessConcept.GROWTH]
            result['formula_type'] = FormulaType.GROWTH_RATE
            result['confidence'] = 0.9
            clues.append("Arithmetic pattern matches growth rate; check header for periods")
            expl.append("Growth rate as (current - previous) / previous")
            matched = True
        # Profit: revenue - cost
        elif '-' in formula_str and any(s for s in src_headers if s and 'revenue' in s.lower()) and any(s for s in src_headers if s and ('cost' in s.lower() or 'expense' in s.lower())):
            result['business_concepts'] = [BusinessConcept.PROFIT]
            result['formula_type'] = FormulaType.CALCULATION
            result['confidence'] = 0.85
            clues.append("Header context = profit calculation as revenue minus cost/expense")
            expl.append("Calculates profit (revenue-cost or revenue-expense)")
            matched = True
        # Variance: actual - budget
        elif '-' in formula_str and any(s for s in src_headers if s and 'actual' in s.lower()) and any(s for s in src_headers if s and 'budget' in s.lower()):
            result['business_concepts'] = [BusinessConcept.VARIANCE]
            result['formula_type'] = FormulaType.CALCULATION
            result['confidence'] = 0.85
            clues.append("Header context = variance calculation as actual minus budget")
            expl.append("Calculates variance (actual-budget)")
            matched = True
        # Conditional (IF): budget/plan versus actual
        elif formula_str.startswith("=IF"):
            result['business_concepts'] = [BusinessConcept.VARIANCE, BusinessConcept.BUDGET, BusinessConcept.ACTUAL]
            result['formula_type'] = FormulaType.CONDITIONAL
            result['confidence'] = 0.8
            clues.append("Conditional logic: often checks budget vs actual status")
            expl.append("Conditional, e.g. IF(Actual > Budget, ...) -> budget-vs-actual comparisons")
            matched = True
        # Ratio: x/y with header cues (e.g., ROE, ROA, ROI)
        elif ('/' in formula_str) and any(s for s in src_headers if s and ('roe' in s.lower() or 'roa' in s.lower() or 'roi' in s.lower())):
            result['business_concepts'] = [BusinessConcept.RATIO]
            result['formula_type'] = FormulaType.RATIO
            result['confidence'] = 0.8
            clues.append("Header context = ratio (ROE/ROA/ROI/etc.)")
            expl.append("Calculates ratio with business header context")
            matched = True
        # SUM: check if header contains revenue, cost, total
        elif formula_str.startswith('=SUM') and any(s for s in [cell_info.row_header, cell_info.column_header] if s and (('revenue' in s.lower()) or ('cost' in s.lower()) or ("total" in s.lower()))):
            if cell_info.row_header and 'revenue' in cell_info.row_header.lower():
                result['business_concepts'] = [BusinessConcept.REVENUE]
            elif cell_info.row_header and 'cost' in cell_info.row_header.lower():
                result['business_concepts'] = [BusinessConcept.COST]
            else:
                result['business_concepts'] = []
            result['formula_type'] = FormulaType.SUM
            result['confidence'] = 0.75
            clues.append("SUM with financial header context")
            expl.append("SUM with business header as context")
            matched = True
        if matched:
            result['context_clues'] = clues
            result['explanation'] = "; ".join(expl)
            return result
        return None
    
    def _convert_gemini_analysis(self, gemini_analysis: GeminiAnalysis, cell_info: CellInfo) -> SemanticInfo:
        """Convert Gemini analysis to SemanticInfo"""
        return SemanticInfo(
            cell_info=cell_info,
            business_concepts=gemini_analysis.business_concepts,
            formula_type=gemini_analysis.formula_type,
            confidence_score=gemini_analysis.confidence_score,
            context_clues=gemini_analysis.context_clues,
            explanation=gemini_analysis.explanation
        )
    
    def _get_row_context(self, cell_info: CellInfo, sheet_context: SheetInfo) -> List[str]:
        """Get context from the same row"""
        context = []
        try:
            if hasattr(sheet_context, 'cells') and sheet_context.cells:
                for cell in sheet_context.cells:
                    if hasattr(cell, 'row') and hasattr(cell, 'col') and hasattr(cell, 'value'):
                        if cell.row == cell_info.row and cell.col != cell_info.col:
                            context.append(str(cell.value) if cell.value else "")
        except Exception as e:
            pass
        return context[:5]  
    
    def _get_column_context(self, cell_info: CellInfo, sheet_context: SheetInfo) -> List[str]:
        """Get context from the same column"""
        context = []
        try:
            if hasattr(sheet_context, 'cells') and sheet_context.cells:
                for cell in sheet_context.cells:
                    if hasattr(cell, 'row') and hasattr(cell, 'col') and hasattr(cell, 'value'):
                        if cell.col == cell_info.col and cell.row != cell_info.row:
                            context.append(str(cell.value) if cell.value else "")
        except Exception as e:
            pass
        return context[:5]  
    
    def _analyze_formula(self, formula: str) -> Tuple[Optional[FormulaType], float]:
        """Analyze the type and meaning of a formula"""
        formula_upper = formula.upper()
        
        for formula_type, pattern_info in self.formula_patterns.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, formula_upper):
                    confidence = 0.9 if formula_type in [FormulaType.SUM, FormulaType.AVERAGE] else 0.7
                    return formula_type, confidence
        
        if '/' in formula and '*' in formula:
            return FormulaType.CALCULATION, 0.6
        elif '/' in formula:
            return FormulaType.RATIO, 0.7
        elif '-' in formula and '(' in formula:
            return FormulaType.GROWTH_RATE, 0.8
        
        return None, 0.0
    
    def _match_business_concepts(self, text: str) -> List[BusinessConcept]:
        """Match business concepts in text"""
        concepts = []
        text_lower = text.lower()
        
        for concept, vocab_info in self.business_vocabulary.items():
            for keyword in vocab_info["keywords"]:
                if keyword.lower() in text_lower:
                    concepts.append(concept)
                    break
            
            for synonym, main_term in vocab_info["synonyms"].items():
                if synonym.lower() in text_lower:
                    concepts.append(concept)
                    break
        
        return concepts
    
    def _analyze_context(self, cell_info: CellInfo, sheet_context: SheetInfo) -> Dict[str, Any]:
        """Analyze context from surrounding cells"""
        clues = []
        confidence = 0.0
        
        nearby_cells = self._get_nearby_cells(cell_info, sheet_context, radius=2)
        
        for nearby_cell in nearby_cells:
            if nearby_cell.value and isinstance(nearby_cell.value, str):
                nearby_text = nearby_cell.value.lower()

                for time_period in self.context_patterns["time_periods"]:
                    if time_period in nearby_text:
                        clues.append(f"Time context: {time_period}")
                        confidence += 0.1

                for unit in self.context_patterns["units"]:
                    if unit in nearby_text:
                        clues.append(f"Unit context: {unit}")
                        confidence += 0.1
        
        return {"clues": clues, "confidence": min(confidence, 0.5)}
    
    def _analyze_headers(self, cell_info: CellInfo, sheet_context: SheetInfo) -> Dict[str, Any]:
        """Analyze headers for additional context"""
        concepts = []
        confidence = 0.0
        
        if cell_info.row <= 3:
            confidence += 0.3
        
        for header in sheet_context.headers:
            header_concepts = self._match_business_concepts(header)
            concepts.extend(header_concepts)
            if header_concepts:
                confidence += 0.2
        
        return {"concepts": concepts, "confidence": min(confidence, 0.4)}
    
    def _get_nearby_cells(self, cell_info: CellInfo, sheet_context: SheetInfo, radius: int = 2) -> List[CellInfo]:
        """Get cells near the target cell"""
        nearby_cells = []
        
        for cell in sheet_context.cells:
            if (abs(cell.row - cell_info.row) <= radius and 
                abs(cell.col - cell_info.col) <= radius and
                cell.cell_address != cell_info.cell_address):
                nearby_cells.append(cell)
        
        return nearby_cells
    
    def _generate_explanation(self, cell_info: CellInfo, concepts: List[BusinessConcept], 
                            formula_type: Optional[FormulaType], context_clues: List[str]) -> str:
        """Generate a human-readable explanation of the cell's semantic meaning"""
        explanation_parts = []
        
        if concepts:
            concept_names = [concept.value for concept in concepts]
            explanation_parts.append(f"Business concepts: {', '.join(concept_names)}")
        
        if formula_type:
            explanation_parts.append(f"Formula type: {formula_type.value}")
        
        if context_clues:
            explanation_parts.append(f"Context: {', '.join(context_clues[:3])}")  # Limit to first 3 clues
        
        if cell_info.is_formula and cell_info.formula:
            explanation_parts.append(f"Formula: {cell_info.formula}")
        
        return "; ".join(explanation_parts) if explanation_parts else "No semantic information identified"
    
    def analyze_sheet_semantics(self, sheet_info: SheetInfo) -> List[SemanticInfo]:
        """Analyze semantic meaning for all cells in a sheet"""
        semantic_results = []
        gemini_cells = []
        rule_based_cells = []
        
        for cell_info in sheet_info.cells:
            if self.use_gemini and self._should_use_gemini(cell_info):
                gemini_cells.append(cell_info)
            else:
                rule_based_cells.append(cell_info)

        for cell_info in rule_based_cells:
            semantic_info = self._analyze_with_rules(cell_info, sheet_info)
            semantic_results.append(semantic_info)
        if gemini_cells and self.use_gemini:
            print(f"Analyzing {len(gemini_cells)} cells with Gemini AI (out of {len(sheet_info.cells)} total)")
            sheet_context_dict = {
                'sheet_name': getattr(sheet_info, 'name', 'Unknown'),
                'headers': self._get_headers_safe(sheet_info),
                'row_context': [],
                'column_context': []
            }
            
            gemini_analyses = self.gemini_service.batch_analyze_cells(gemini_cells, sheet_context_dict)
            
            for i, (cell_info, gemini_analysis) in enumerate(zip(gemini_cells, gemini_analyses)):
                if gemini_analysis:
                    semantic_info = self._convert_gemini_analysis(gemini_analysis, cell_info)
                else:
                    semantic_info = self._analyze_with_rules(cell_info, sheet_info)
                semantic_results.append(semantic_info)
        
        return semantic_results
    
    def get_concept_summary(self, semantic_results: List[SemanticInfo]) -> Dict[BusinessConcept, List[SemanticInfo]]:
        """Group semantic results by business concept"""
        concept_groups = {}
        
        for semantic_info in semantic_results:
            for concept in semantic_info.business_concepts:
                if concept not in concept_groups:
                    concept_groups[concept] = []
                concept_groups[concept].append(semantic_info)
        
        return concept_groups



if __name__ == "__main__":
    engine = SemanticEngine()
    test_texts = [
        "Q1 Revenue",
        "Gross Profit Margin %",
        "Budget vs Actual Variance",
        "Year over Year Growth",
        "ROI Calculation"
    ]
    
    print("Testing business concept recognition:")
    for text in test_texts:
        concepts = engine._match_business_concepts(text)
        print(f"'{text}' -> {[c.value for c in concepts]}")
    
    print("\nSemanticEngine initialized successfully!")
    print("Ready to analyze spreadsheet semantics.")
