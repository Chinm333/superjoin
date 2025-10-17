"""
Main Application for Semantic Spreadsheet Search Engine

This module provides the main interface for the semantic search system,
combining all components to provide a complete search experience.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path
try:
    from .parser import SpreadsheetParser, SheetInfo
    from .semantic_engine import SemanticEngine, SemanticInfo
    from .query_processor import QueryProcessor, QueryIntent
    from .ranking import ResultRanker, ResultFormatter, SearchResult, ResultFormat
    from .answer_generator import AnswerGenerator
    from .config import Config
except ImportError:
    from parser import SpreadsheetParser, SheetInfo
    from semantic_engine import SemanticEngine, SemanticInfo
    from query_processor import QueryProcessor, QueryIntent
    from ranking import ResultRanker, ResultFormatter, SearchResult, ResultFormat
    from answer_generator import AnswerGenerator
    from config import Config


class SemanticSpreadsheetSearch:
    """Main class for semantic spreadsheet search functionality"""
    
    def __init__(self):
        # Create shared Gemini service instance to avoid quota issues
        try:
            from .gemini_service import GeminiService
        except ImportError:
            from gemini_service import GeminiService
        
        self.shared_gemini_service = GeminiService()
        
        self.parser = SpreadsheetParser()
        self.semantic_engine = SemanticEngine()
        self.query_processor = QueryProcessor()
        self.ranker = ResultRanker()
        self.formatter = ResultFormatter()
        self.answer_generator = AnswerGenerator()
        
        # Share the Gemini service instance across all components
        self.semantic_engine.gemini_service = self.shared_gemini_service
        self.query_processor.gemini_service = self.shared_gemini_service
        self.ranker.gemini_service = self.shared_gemini_service
        self.answer_generator.gemini_service = self.shared_gemini_service
        
        self.loaded_sheets: Dict[str, SheetInfo] = {}
        self.semantic_cache: Dict[str, List[SemanticInfo]] = {}
        
        # Show integration status
        self._show_integration_status()
    
    def _show_integration_status(self):
        """Show the status of AI integrations"""
        
        print("\n" + "="*60)
        print("SEMANTIC SPREADSHEET SEARCH ENGINE")
        print("="*60)
        
        if Config.USE_GEMINI:
            print("Gemini AI: ENABLED - Enhanced semantic understanding")
        else:
            print("Gemini AI: DISABLED - Using rule-based analysis")
            print("   Run 'python setup_gemini.py' to enable AI features")
        
        print("="*60)
    
    def load_spreadsheet(self, file_path: str) -> Dict[str, Any]:
        """
        Load and analyze a spreadsheet file
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with loading results and sheet summaries
        """
        try:
            # Parse the spreadsheet
            sheets_info = self.parser.load_excel_file(file_path)
            self.loaded_sheets.update(sheets_info)
            
            # Perform semantic analysis on all sheets
            for sheet_name, sheet_info in sheets_info.items():
                semantic_results = self.semantic_engine.analyze_sheet_semantics(sheet_info)
                self.semantic_cache[sheet_name] = semantic_results
            
            # Generate summaries
            summaries = {}
            for sheet_name in sheets_info.keys():
                summaries[sheet_name] = self.parser.get_sheet_summary(sheet_name)
            
            return {
                "success": True,
                "file_path": file_path,
                "sheets_loaded": list(sheets_info.keys()),
                "total_sheets": len(sheets_info),
                "summaries": summaries
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def search(self, query: str, sheet_name: Optional[str] = None, 
              format_type: ResultFormat = ResultFormat.HUMAN_READABLE,
              max_results: int = 20) -> str:
        """
        Perform semantic search on loaded spreadsheets
        
        Args:
            query: Natural language search query
            sheet_name: Optional specific sheet to search (searches all if None)
            format_type: Output format for results
            max_results: Maximum number of results to return
            
        Returns:
            Formatted search results
        """
        start_time = time.time()
        
        # Validate query
        is_valid, validation_message = self.query_processor.validate_query(query)
        if not is_valid:
            return f"Invalid query: {validation_message}"
        
        # Process the query
        query_intent = self.query_processor.process_query(query)
        
        # Get semantic results to search
        semantic_results = self._get_semantic_results(sheet_name)
        
        if not semantic_results:
            return "No spreadsheet data loaded. Please load a spreadsheet first."
        
        # Rank the results
        search_results = self.ranker.rank_results(semantic_results, query_intent)
        
        # Limit results
        search_results = search_results[:max_results]
        
        # Format and return results
        search_time = time.time() - start_time
        
        # Create SearchResults object for formatting
        try:
            from .ranking import SearchResults
        except ImportError:
            from ranking import SearchResults
        search_results_obj = SearchResults(
            results=search_results,
            total_found=len(search_results),
            query_intent=query_intent,
            search_time=search_time,
            format_type=format_type
        )
        
        return self.formatter.format_results(search_results, query_intent, format_type)
    
    def search_with_answer(self, query: str, sheet_name: Optional[str] = None, 
                          format_type: ResultFormat = ResultFormat.HUMAN_READABLE,
                          max_results: int = 20, include_answer: bool = True) -> str:
        """
        Perform semantic search and generate direct answers
        
        Args:
            query: Natural language search query
            sheet_name: Optional specific sheet to search (searches all if None)
            format_type: Output format for results
            max_results: Maximum number of results to return
            include_answer: Whether to include direct answer generation
            
        Returns:
            Formatted search results with direct answer
        """
        start_time = time.time()
        
        # Validate query
        is_valid, validation_message = self.query_processor.validate_query(query)
        if not is_valid:
            return f"Invalid query: {validation_message}"
        
        # Process the query
        query_intent = self.query_processor.process_query(query)
        
        # Get semantic results to search
        semantic_results = self._get_semantic_results(sheet_name)
        
        if not semantic_results:
            return "No spreadsheet data loaded. Please load a spreadsheet first."
        
        # Rank the results
        search_results = self.ranker.rank_results(semantic_results, query_intent)
        
        # Limit results
        search_results = search_results[:max_results]
        
        # Format search results
        search_time = time.time() - start_time
        search_output = self.formatter.format_results(search_results, query_intent, format_type)
        
        # Generate direct answer if requested
        if include_answer and search_results:
            try:
                answer = self.answer_generator.generate_answer(search_results, query_intent)
                if answer and answer.strip():
                    return f"{search_output}\n\n{answer}"
            except Exception as e:
                print(f"Answer generation failed: {e}")
                # Return just the search results if answer generation fails
        
        return search_output
    
    def _get_semantic_results(self, sheet_name: Optional[str] = None) -> List[SemanticInfo]:
        """Get semantic results for search"""
        if sheet_name:
            return self.semantic_cache.get(sheet_name, [])
        else:
            # Combine all sheets
            all_results = []
            for sheet_results in self.semantic_cache.values():
                all_results.extend(sheet_results)
            return all_results
    
    def get_sheet_info(self, sheet_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific sheet"""
        if sheet_name not in self.loaded_sheets:
            return None
        
        sheet_info = self.loaded_sheets[sheet_name]
        semantic_results = self.semantic_cache.get(sheet_name, [])
        
        # Analyze concepts in the sheet
        concept_summary = self.semantic_engine.get_concept_summary(semantic_results)
        
        return {
            "name": sheet_name,
            "total_cells": len(sheet_info.cells),
            "formulas": len(sheet_info.formulas),
            "headers": sheet_info.headers,
            "data_range": sheet_info.data_range,
            "concepts_found": {concept.value: len(results) for concept, results in concept_summary.items()},
            "formula_types": self.parser._analyze_formula_types(sheet_info.formulas)
        }
    
    def list_loaded_sheets(self) -> List[str]:
        """Get list of loaded sheet names"""
        return list(self.loaded_sheets.keys())
    
    def get_concept_overview(self) -> Dict[str, Any]:
        """Get overview of all business concepts found across all sheets"""
        all_concepts = {}
        
        for sheet_name, semantic_results in self.semantic_cache.items():
            concept_summary = self.semantic_engine.get_concept_summary(semantic_results)
            
            for concept, results in concept_summary.items():
                concept_name = concept.value
                if concept_name not in all_concepts:
                    all_concepts[concept_name] = {
                        "total_occurrences": 0,
                        "sheets": {}
                    }
                
                all_concepts[concept_name]["total_occurrences"] += len(results)
                all_concepts[concept_name]["sheets"][sheet_name] = len(results)
        
        return all_concepts
    
    def suggest_queries(self, concept: Optional[str] = None) -> List[str]:
        """Suggest example queries based on loaded data"""
        suggestions = []
        
        if concept:
            # Suggest queries for a specific concept
            concept_suggestions = {
                "revenue": [
                    "find all revenue calculations",
                    "show total sales formulas",
                    "where is the revenue data?"
                ],
                "profit": [
                    "find profitability metrics",
                    "show profit calculations",
                    "where are the margin analyses?"
                ],
                "cost": [
                    "find all cost calculations",
                    "show expense formulas",
                    "where are the spending analyses?"
                ],
                "growth": [
                    "find growth rate calculations",
                    "show year over year changes",
                    "where are the trend analyses?"
                ]
            }
            
            suggestions.extend(concept_suggestions.get(concept.lower(), []))
        else:
            # General suggestions
            suggestions = [
                "find all revenue calculations",
                "show profitability metrics",
                "where are my margin analyses?",
                "find budget vs actual comparisons",
                "show efficiency ratios",
                "find all percentage calculations",
                "where is the Q1 data?",
                "show cost-related formulas"
            ]
        
        return suggestions
    
    def export_results(self, query: str, output_format: str = "json", 
                      file_path: Optional[str] = None) -> str:
        """
        Export search results to a file
        
        Args:
            query: Search query
            output_format: Output format ("json", "csv", "txt")
            file_path: Optional file path (auto-generates if None)
            
        Returns:
            Path to exported file
        """
        # Perform search
        if output_format.lower() == "json":
            results = self.search(query, format_type=ResultFormat.STRUCTURED)
        else:
            results = self.search(query, format_type=ResultFormat.DETAILED)
        
        # Generate file path if not provided
        if not file_path:
            timestamp = int(time.time())
            file_path = f"search_results_{timestamp}.{output_format}"
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(results)
        
        return file_path


def main():
    """Main function for interactive chat interface"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Semantic Spreadsheet Search Engine - Interactive Chat")
    parser.add_argument("--file", "-f", help="Excel file to load initially")
    parser.add_argument("--format", choices=["human", "json", "grouped", "detailed"], 
                       default="human", help="Output format")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum results")
    
    args = parser.parse_args()
    
    search_engine = SemanticSpreadsheetSearch()
    
    if args.file:
        result = search_engine.load_spreadsheet(args.file)
        if result["success"]:
            print(f"✅ Loaded {result['total_sheets']} sheets from {args.file}")
            for sheet_name, summary in result["summaries"].items():
                print(f"   📊 {sheet_name}: {summary['total_cells']} cells, {summary['formula_count']} formulas")
        else:
            print(f"❌ Error loading file: {result['error']}")
            return
    start_chat_interface(search_engine, args.format, args.max_results)


def start_chat_interface(search_engine, default_format, max_results):
    """Start the interactive chat interface"""
    
    print("\n" + "="*70)
    print("🤖 SEMANTIC SPREADSHEET SEARCH - INTERACTIVE CHAT")
    print("="*70)
    print("💬 Ask questions about your Excel files in natural language!")
    print("📁 Load files with: load <file_path>")
    print("❓ Type 'help' for commands or 'exit' to quit")
    print("="*70)
    
    if search_engine.list_loaded_sheets():
        print(f"📋 Currently loaded: {', '.join(search_engine.list_loaded_sheets())}")
    else:
        print("📋 No files loaded yet. Use 'load <file_path>' to get started!")
    
    print("\n" + "💬 Chat started! Ask me anything about your spreadsheets:")
    
    while True:
        try:
            user_input = input("\n🔍 You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n👋 Thanks for using Semantic Spreadsheet Search! Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                show_help_menu()
                continue
            
            elif user_input.lower().startswith('load '):
                file_path = user_input[5:].strip()
                handle_load_command(search_engine, file_path)
                continue
            
            elif user_input.lower() == 'sheets':
                handle_sheets_command(search_engine)
                continue
            
            elif user_input.lower().startswith('info '):
                sheet_name = user_input[5:].strip()
                handle_info_command(search_engine, sheet_name)
                continue
            
            elif user_input.lower() == 'concepts':
                handle_concepts_command(search_engine)
                continue
            
            elif user_input.lower() == 'suggest':
                handle_suggest_command(search_engine)
                continue
            
            elif user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print("🧹 Screen cleared!")
                continue
            
            else:
                handle_search_query(search_engine, user_input, default_format, max_results)
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Semantic Spreadsheet Search! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Try typing 'help' for available commands or check your input.")


def show_help_menu():
    """Display the help menu"""
    print("\n" + "="*50)
    print("📚 HELP MENU - Available Commands")
    print("="*50)
    print("🔍 SEARCH COMMANDS:")
    print("   • Just type your question naturally!")
    print("   • Examples: 'find all revenue calculations', 'show profit margins'")
    print()
    print("📁 FILE COMMANDS:")
    print("   • load <file_path>     - Load an Excel file")
    print("   • sheets               - List all loaded sheets")
    print("   • info <sheet_name>    - Get detailed sheet information")
    print()
    print("📊 DATA COMMANDS:")
    print("   • concepts             - Show all business concepts found")
    print("   • suggest              - Get query suggestions")
    print()
    print("🛠️  UTILITY COMMANDS:")
    print("   • clear                - Clear the screen")
    print("   • help                 - Show this help menu")
    print("   • exit/quit            - Exit the application")
    print("="*50)


def handle_load_command(search_engine, file_path):
    """Handle file loading command"""
    import os
    
    if not file_path:
        print("❌ Please provide a file path. Usage: load <file_path>")
        return
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"🔄 Loading file: {file_path}")
    
    try:
        result = search_engine.load_spreadsheet(file_path)
        if result["success"]:
            print(f"✅ Successfully loaded {result['total_sheets']} sheets!")
            for sheet_name, summary in result["summaries"].items():
                print(f"   📊 {sheet_name}: {summary['total_cells']} cells, {summary['formula_count']} formulas")
            print("💬 You can now ask questions about this data!")
        else:
            print(f"❌ Error loading file: {result['error']}")
    except Exception as e:
        print(f"❌ Error: {e}")


def handle_sheets_command(search_engine):
    """Handle sheets listing command"""
    sheets = search_engine.list_loaded_sheets()
    if sheets:
        print(f"\n📋 Loaded sheets ({len(sheets)}):")
        for i, sheet in enumerate(sheets, 1):
            print(f"   {i}. {sheet}")
    else:
        print("📋 No sheets loaded. Use 'load <file_path>' to load an Excel file.")


def handle_info_command(search_engine, sheet_name):
    """Handle sheet info command"""
    if not sheet_name:
        print("❌ Please provide a sheet name. Usage: info <sheet_name>")
        return
    
    info = search_engine.get_sheet_info(sheet_name)
    if info:
        print(f"\n📊 Sheet Information: {info['name']}")
        print("="*40)
        print(f"📝 Total Cells: {info['total_cells']}")
        print(f"🧮 Formulas: {info['formulas']}")
        print(f"📋 Headers: {len(info['headers'])} found")
        print(f"🎯 Data Range: {info['data_range']}")
        
        if info['concepts_found']:
            print(f"\n💡 Business Concepts Found:")
            for concept, count in info['concepts_found'].items():
                print(f"   • {concept}: {count} occurrences")
        
        if info['formula_types']:
            print(f"\n🧮 Formula Types:")
            for ftype, count in info['formula_types'].items():
                print(f"   • {ftype}: {count} formulas")
    else:
        print(f"❌ Sheet '{sheet_name}' not found. Use 'sheets' to see available sheets.")


def handle_concepts_command(search_engine):
    """Handle concepts overview command"""
    concepts = search_engine.get_concept_overview()
    if concepts:
        print(f"\n💡 Business Concepts Found:")
        print("="*40)
        for concept, data in concepts.items():
            print(f"📊 {concept}: {data['total_occurrences']} total occurrences")
            if data['sheets']:
                for sheet, count in data['sheets'].items():
                    print(f"   └─ {sheet}: {count} occurrences")
    else:
        print("💡 No business concepts found. Load an Excel file first.")


def handle_suggest_command(search_engine):
    """Handle query suggestions command"""
    suggestions = search_engine.suggest_queries()
    if suggestions:
        print(f"\n💡 Query Suggestions:")
        print("="*40)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        print("\n💬 Try copying any of these questions!")
    else:
        print("💡 No suggestions available. Load an Excel file first.")


def handle_search_query(search_engine, query, format_type, max_results, use_answer_generator=True):
    """Handle natural language search queries"""
    if not search_engine.list_loaded_sheets():
        print("❌ No files loaded. Please load an Excel file first using 'load <file_path>'")
        return
    
    print(f"\n🔍 Searching for: '{query}'")
    print("⏳ Processing...")
    
    try:
        # Use the new search_with_answer method for enhanced results
        if use_answer_generator:
            results = search_engine.search_with_answer(query, format_type=format_type, max_results=max_results)
        else:
            results = search_engine.search(query, format_type=format_type, max_results=max_results)
        
        if results and results.strip():
            print(f"\n🤖 Results:")
            print("="*50)
            print(results)
            print("="*50)
        else:
            print("\n🤖 No results found for your query.")
            print("💡 Try rephrasing your question or use 'suggest' for ideas.")
            
    except Exception as e:
        print(f"\n❌ Search error: {e}")
        print("💡 Try rephrasing your query or check if files are loaded properly.")


if __name__ == "__main__":
    main()
