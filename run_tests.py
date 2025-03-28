#!/usr/bin/env python
"""Run tests for document_impact_analyzer and print detailed results"""

import os
import sys
import unittest
import importlib

def run_tests():
    """Run all tests and print results"""
    print("Running tests for document_impact_analyzer package...")
    print("=" * 60)
    
    # Make sure the package is imported properly
    try:
        import document_impact_analyzer
        print(f"Successfully imported package version: {document_impact_analyzer.__version__}")
    except ImportError as e:
        print(f"Error importing package: {e}")
        sys.exit(1)
    
    # Get the tests directory
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    if not os.path.exists(test_dir):
        print(f"Error: Tests directory not found at {test_dir}")
        sys.exit(1)
    
    print(f"Found tests directory: {test_dir}")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    tests = loader.discover(test_dir)
    
    # Run the tests with detailed output
    print("\nTest Results:")
    print("-" * 60)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    
    # Print summary
    print("\nTest Summary:")
    print(f"  Ran {result.testsRun} tests")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    # Print any failures or errors
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)