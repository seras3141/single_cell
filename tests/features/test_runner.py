#!/usr/bin/env python3
"""
Test runner script for feature extraction tests.

This script provides an easy way to run different test suites for the
feature extraction functionality.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type, verbose=False, coverage=False):
    """Run specified test suite.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Whether to run tests in verbose mode
        coverage: Whether to include coverage reporting
    """
    base_cmd = ['python', '-m', 'pytest']
    
    # Add coverage if requested
    if coverage:
        base_cmd.extend(['--cov=src/features', '--cov-report=html', '--cov-report=term'])
    
    # Add verbose flag if requested
    if verbose:
        base_cmd.append('-v')
    else:
        base_cmd.append('--tb=short')
    
    # Select test files based on type
    project_root = Path(__file__).parent.parent.parent  # Go up to project root
    
    if test_type == 'unit':
        test_path = str(project_root / 'tests/features/test_feature_extractor_2d.py')
    elif test_type == 'integration':
        test_path = str(project_root / 'tests/features/test_integration.py')
    elif test_type == 'all':
        test_path = str(project_root / 'tests/features/')
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    base_cmd.append(test_path)
    
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(base_cmd)}")
    print("-" * 50)
    
    # Run the tests
    try:
        result = subprocess.run(base_cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run feature extraction tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py unit              # Run unit tests only
  python test_runner.py integration -v    # Run integration tests with verbose output
  python test_runner.py all --coverage    # Run all tests with coverage
        """
    )
    
    parser.add_argument(
        'test_type',
        choices=['unit', 'integration', 'all'],
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Run tests in verbose mode'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Include coverage reporting'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Run tests with minimal output (opposite of verbose)'
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent.parent  # Go up to project root
    original_dir = Path.cwd()
    
    try:
        # Change to project directory for running tests
        import os
        os.chdir(str(project_root))
        
        # Run tests
        success = run_tests(
            test_type=args.test_type,
            verbose=args.verbose and not args.fast,
            coverage=args.coverage
        )
        
        # Print summary
        print("-" * 50)
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
        return 0 if success else 1
        
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == '__main__':
    sys.exit(main())
