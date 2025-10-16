"""
Type definitions and enums for the semantic spreadsheet search engine
"""

from enum import Enum


class BusinessConcept(Enum):
    """Business concepts that can be recognized in spreadsheets"""
    REVENUE = "revenue"
    COST = "cost"
    PROFIT = "profit"
    MARGIN = "margin"
    RATIO = "ratio"
    GROWTH = "growth"
    EFFICIENCY = "efficiency"
    BUDGET = "budget"
    ACTUAL = "actual"
    FORECAST = "forecast"
    PERCENTAGE = "percentage"
    AVERAGE = "average"
    TOTAL = "total"
    VARIANCE = "variance"
    COMPARISON = "comparison"
    TREND = "trend"
    BENCHMARK = "benchmark"


class FormulaType(Enum):
    """Types of formulas that can be recognized"""
    SUM = "sum"
    AVERAGE = "average"
    COUNT = "count"
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    GROWTH_RATE = "growth_rate"
    CONDITIONAL = "conditional"
    LOOKUP = "lookup"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
