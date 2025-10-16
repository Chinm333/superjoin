# Semantic Spreadsheet Search Engine

A revolutionary search engine that understands spreadsheet content conceptually and allows users to find what they're looking for using natural language queries.

## 🎯 The Problem We Solve

**Traditional spreadsheet search is limited:**
- ❌ Only finds exact text matches
- ❌ Doesn't understand business concepts
- ❌ Requires knowing exact cell locations
- ❌ No context about what formulas actually do

**Our solution provides:**
- ✅ **Semantic Understanding**: Recognizes that "Q1 Revenue", "First Quarter Sales", and "Jan-Mar Income" are the same concept
- ✅ **Natural Language Queries**: "Find all profitability metrics" returns gross margin, net profit, EBITDA calculations
- ✅ **Business Context**: Understands that =B5/B6 in a "Margin %" column calculates a margin
- ✅ **Intelligent Results**: Returns meaningful explanations, not just cell references

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd semantic-spreadsheet-search

# Install dependencies
pip install -r requirements.txt

# Setup Gemini AI integration (optional but recommended)
python setup_gemini.py
```

### Basic Usage
```python
from src.main import SemanticSpreadsheetSearch

# Initialize the search engine
search_engine = SemanticSpreadsheetSearch()

# Load a spreadsheet
result = search_engine.load_spreadsheet("financial_model.xlsx")
print(f"Loaded {result['total_sheets']} sheets")

# Search with natural language
results = search_engine.search("find all margin calculations")
print(results)
```

### Interactive Chat Interface 🎉

**NEW!** Experience the most intuitive way to interact with your spreadsheets:

```bash
# Start the interactive chat
python src/main.py

# Or with an initial file
python src/main.py --file your_spreadsheet.xlsx
```

**Chat Features:**
- 💬 **Natural Conversation**: Ask questions like "What are my profit margins?" or "Show me all revenue calculations"
- 📁 **Dynamic File Loading**: Load Excel files anytime during the chat with `load <file_path>`
- 🔍 **Smart Search**: Get intelligent results with explanations
- 📊 **Data Exploration**: Use commands like `sheets`, `concepts`, `suggest` to explore your data
- 🛠️ **Help System**: Type `help` for available commands
- 🚪 **Easy Exit**: Type `exit` or `quit` to end the session

**Example Chat Session:**
```
🤖 SEMANTIC SPREADSHEET SEARCH - INTERACTIVE CHAT
======================================================================
💬 Ask questions about your Excel files in natural language!
📁 Load files with: load <file_path>
❓ Type 'help' for commands or 'exit' to quit
======================================================================

🔍 You: load SalesDashboard.xlsx
✅ Successfully loaded 3 sheets!
   📊 Q1_Sales: 150 cells, 25 formulas
   📊 Q2_Sales: 180 cells, 30 formulas
   📊 Summary: 45 cells, 15 formulas

🔍 You: find all revenue calculations
🔍 Searching for: 'find all revenue calculations'
⏳ Processing...

🤖 Results:
==================================================
Found 12 revenue-related calculations across 3 sheets:
• Q1_Sales!C15: Total Q1 Revenue = $125,000
• Q2_Sales!C15: Total Q2 Revenue = $145,000
• Summary!B5: Year-to-Date Revenue = $270,000
...
==================================================

🔍 You: exit
👋 Thanks for using Semantic Spreadsheet Search! Goodbye!
```

### Command Line Interface
```bash
# Load a file and search
python src/main.py --file financial_model.xlsx --query "find profitability metrics"

# Interactive mode
python src/main.py --file financial_model.xlsx
```

### Run Demo
```bash
# Non-interactive demo with sample data
python demo/demo_non_interactive.py

# Interactive demo
python demo/demo.py
```

## 📊 Example Queries & Results

### Query: "find all revenue calculations"
**Returns:**
- Revenue total formulas (SUM functions)
- Revenue percentage calculations
- Revenue growth rate formulas
- Budget vs actual revenue comparisons

### Query: "show profitability metrics"
**Returns:**
- Gross profit calculations
- Operating profit formulas
- Margin percentage calculations
- Profit trend analyses

### Query: "find budget vs actual comparisons"
**Returns:**
- Variance calculations
- Percentage differences
- Comparison formulas
- Budget analysis sections

## 🏗️ Architecture

### Core Components

1. **Spreadsheet Parser** (`src/parser.py`)
   - Extracts formulas, values, headers, and cell relationships
   - Supports Excel (.xlsx) and Google Sheets
   - Identifies data structure and context

2. **Semantic Engine** (`src/semantic_engine.py`) 🧠
   - **Gemini AI Enhanced**: Dynamic business concept recognition
   - Recognizes 15+ business concepts (revenue, cost, profit, margin, etc.)
   - Understands formula semantics and business meaning
   - Analyzes context from surrounding cells
   - **Fallback**: Rule-based analysis if AI unavailable

3. **Query Processor** (`src/query_processor.py`) 💬
   - **Gemini AI Enhanced**: Advanced natural language understanding
   - Processes natural language queries with AI-powered intent recognition
   - Extracts business concepts and search criteria
   - Classifies query types (conceptual, functional, comparative)
   - **Fallback**: Pattern-based processing if AI unavailable

4. **Ranking System** (`src/ranking.py`) 🎯
   - **Gemini AI Enhanced**: Intelligent result explanations
   - Multi-factor relevance scoring
   - Multiple output formats (human-readable, JSON, grouped)
   - Contextual explanations for results
   - **Fallback**: Rule-based explanations if AI unavailable

5. **Gemini Service** (`src/gemini_service.py`) 🤖
   - **NEW**: AI integration layer
   - Handles all Gemini API interactions
   - Batch processing for efficiency
   - Error handling and rate limiting
   - Graceful fallback management

### System Flow
```
User Query → Query Processor → Semantic Engine → Ranking System → Formatted Results
     ↓              ↓              ↓              ↓
Natural Language → Business Concepts → Relevance Scoring → Meaningful Output
     ↓              ↓              ↓              ↓
🧠 Gemini AI → 🧠 Gemini AI → 🧠 Gemini AI → Enhanced Understanding
```

### AI Integration Status
The system shows integration status on startup:
- 🧠 **Gemini AI: ENABLED** - Enhanced semantic understanding
- 📋 **Gemini AI: DISABLED** - Using rule-based analysis

## 🧪 Testing

### Run Basic Tests
```bash
python tests/test_basic.py
```

### Test Individual Components
```python
# Test parser
from src.parser import SpreadsheetParser
parser = SpreadsheetParser()

# Test semantic engine  
from src.semantic_engine import SemanticEngine
engine = SemanticEngine()

# Test query processor
from src.query_processor import QueryProcessor
processor = QueryProcessor()
```

## 📁 Project Structure

```
semantic-spreadsheet-search/
├── src/                          # Core source code
│   ├── __init__.py
│   ├── parser.py                 # Spreadsheet content parser
│   ├── semantic_engine.py        # Business concept recognition
│   ├── query_processor.py        # Natural language query processing
│   ├── ranking.py                # Result ranking and formatting
│   └── main.py                   # Main application interface
├── tests/                        # Test files
│   └── test_basic.py            # Basic component tests
├── demo/                         # Demo applications
│   ├── demo.py                  # Interactive demo
│   └── demo_non_interactive.py  # Automated demo
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── DESIGN_DOCUMENT.md           # Technical design document
```

## 🔧 Configuration

### Supported Business Concepts
- **Financial**: Revenue, Cost, Profit, Margin, Ratio
- **Operational**: Growth, Efficiency, Budget, Actual, Variance
- **Analytical**: Percentage, Average, Total, Comparison, Trend

### Supported Formula Types
- **Aggregation**: SUM, AVERAGE, COUNT
- **Calculation**: Percentage, Ratio, Growth Rate
- **Logic**: Conditional (IF), Lookup (VLOOKUP)
- **Comparison**: Variance, Difference

### Output Formats
- **Human Readable**: Natural language summaries
- **Structured JSON**: Programmatic access
- **Grouped by Concept**: Organized by business concept
- **Detailed**: Full context and explanations

## 🚀 Advanced Usage

### Custom Business Concepts
```python
from src.semantic_engine import BusinessConcept

# Add custom concepts (extend the enum)
class CustomBusinessConcept(BusinessConcept):
    CUSTOM_METRIC = "custom_metric"
```

### Batch Processing
```python
# Process multiple files
files = ["model1.xlsx", "model2.xlsx", "model3.xlsx"]
for file in files:
    search_engine.load_spreadsheet(file)
    results = search_engine.search("find all revenue calculations")
    print(f"Results for {file}: {len(results)} found")
```

### Export Results
```python
# Export to different formats
search_engine.export_results("find profitability metrics", "json", "results.json")
search_engine.export_results("show cost calculations", "csv", "costs.csv")
```

## 🎯 Use Cases

### Financial Analysis
- Find all revenue calculations across multiple sheets
- Locate profitability metrics and margin analyses
- Identify budget vs actual variance calculations

### Business Intelligence
- Discover efficiency ratios and performance metrics
- Find growth rate calculations and trend analyses
- Locate comparative analyses and benchmarks

### Data Auditing
- Verify formula consistency across sheets
- Find all percentage calculations for review
- Locate conditional logic and lookup formulas

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built for the semantic search challenge
- Inspired by the need for more intuitive spreadsheet tools
- Thanks to the open-source community for foundational libraries

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the design document for technical details
- Run the demo to see the system in action

---

**Transform how you search spreadsheets - from structural to semantic!** 🎯

## step1
python setup_gemini.py
## step2
# Demo with AI enhancement
python demo/demo_non_interactive.py

# Interactive mode
python demo/demo.py

# Command line
python src/main.py --file your_file.xlsx --query "find profitability metrics"