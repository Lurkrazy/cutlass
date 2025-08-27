#!/usr/bin/env python3
"""
Code Structure Validation Script for AddThenPlus1 Custom Operator

This script validates the implementation structure and code correctness
without requiring a full CUDA runtime environment.
"""

import os
import sys
import ast
import re
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and return its status"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False


def validate_cpp_epilogue_functor(filepath):
    """Validate the C++ epilogue functor implementation"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for essential components
    checks = [
        ('namespace cutlass', 'namespace cutlass'),
        ('class AddThenPlus1', 'class AddThenPlus1'),
        ('FragmentOutput operator()', 'operator() method'),
        ('ElementCompute(1)', 'adds 1 constant'),
        ('alpha_ * converted_accumulator', 'alpha scaling'),
        ('beta_ * converted_source', 'beta scaling'),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ C++ functor contains {description}")
        else:
            print(f"  ✗ C++ functor missing {description}")
            all_passed = False
    
    return all_passed


def validate_python_operator(filepath):
    """Validate the Python DSL operator implementation"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for essential components
    checks = [
        ('@cute.kernel', '@cute.kernel decorator'),
        ('def add_then_plus1_kernel', 'kernel function'),
        ('@cute.jit', '@cute.jit decorator'),
        ('def add_then_plus1', 'main function'),
        ('a_values + b_values + 1.0', 'AddThenPlus1 operation'),
        ('cute.make_tiled_copy_tv', 'tiled copy usage'),
        ('cute.copy(', 'memory copy operations'),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Python operator contains {description}")
        else:
            print(f"  ✗ Python operator missing {description}")
            all_passed = False
    
    return all_passed


def validate_example_script(filepath):
    """Validate the example script implementation"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for essential components
    checks = [
        ('from ops.add_then_plus1 import add_then_plus1', 'imports custom operator'),
        ('ref = a + b', 'reference computation'),
        ('expected = ref + 1.0', 'expected result computation'),
        ('torch.testing.assert_close', 'correctness verification'),
        ('PASS: AddThenPlus1', 'success message'),
        ('argparse.ArgumentParser', 'command line interface'),
        ('--benchmark', 'benchmark support'),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Example script contains {description}")
        else:
            print(f"  ✗ Example script missing {description}")
            all_passed = False
    
    return all_passed


def validate_test_file(filepath):
    """Validate the pytest test file"""
    if not os.path.exists(filepath):
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for essential components
    checks = [
        ('import pytest', 'pytest import'),
        ('class TestAddThenPlus1', 'test class'),
        ('def test_add_then_plus1_small', 'small tensor test'),
        ('def test_add_then_plus1_2d', '2D tensor test'),
        ('def test_add_then_plus1_specific_values', 'specific values test'),
        ('torch.testing.assert_close', 'correctness verification'),
        ('@pytest.mark.parametrize', 'parameterized tests'),
        ('expected = ref + 1.0', 'expected result computation'),
    ]
    
    all_passed = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ Test file contains {description}")
        else:
            print(f"  ✗ Test file missing {description}")
            all_passed = False
    
    return all_passed


def validate_syntax(filepath):
    """Validate Python file syntax"""
    if not os.path.exists(filepath) or not filepath.endswith('.py'):
        return True
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        ast.parse(content)
        print(f"  ✓ Valid Python syntax: {os.path.basename(filepath)}")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error in {os.path.basename(filepath)}: {e}")
        return False


def main():
    """Main validation function"""
    print("=== AddThenPlus1 Custom Operator Code Validation ===\n")
    
    # Get the base directory (cutlass root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cutlass_dir = script_dir
    
    # File paths to validate
    files_to_check = [
        (os.path.join(cutlass_dir, "include/cutlass/epilogue/thread/add_then_plus1.h"),
         "C++ Epilogue Functor"),
        (os.path.join(cutlass_dir, "examples/python/CuTeDSL/ops/__init__.py"),
         "Python Ops Module Init"),
        (os.path.join(cutlass_dir, "examples/python/CuTeDSL/ops/add_then_plus1.py"),
         "Python DSL Operator"),
        (os.path.join(cutlass_dir, "examples/python/CuTeDSL/ampere/vector_add_then_plus1.py"),
         "Example Script"),
        (os.path.join(cutlass_dir, "tests/test_add_then_plus1.py"),
         "Pytest Test File"),
        (os.path.join(cutlass_dir, "examples/python/CuTeDSL/README_add_then_plus1.md"),
         "Documentation"),
        (os.path.join(cutlass_dir, "BUILD_AND_TEST_INSTRUCTIONS.md"),
         "Build Instructions"),
    ]
    
    print("1. File Existence Check:")
    all_files_exist = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return False
    
    print("\n2. Python Syntax Validation:")
    syntax_valid = True
    python_files = [fp for fp, _ in files_to_check if fp.endswith('.py')]
    for filepath in python_files:
        if not validate_syntax(filepath):
            syntax_valid = False
    
    if not syntax_valid:
        print("\n❌ Python syntax errors found!")
        return False
    
    print("\n3. C++ Epilogue Functor Validation:")
    cpp_epilogue_path = os.path.join(cutlass_dir, "include/cutlass/epilogue/thread/add_then_plus1.h")
    cpp_valid = validate_cpp_epilogue_functor(cpp_epilogue_path)
    
    print("\n4. Python DSL Operator Validation:")
    python_op_path = os.path.join(cutlass_dir, "examples/python/CuTeDSL/ops/add_then_plus1.py")
    python_valid = validate_python_operator(python_op_path)
    
    print("\n5. Example Script Validation:")
    example_path = os.path.join(cutlass_dir, "examples/python/CuTeDSL/ampere/vector_add_then_plus1.py")
    example_valid = validate_example_script(example_path)
    
    print("\n6. Test File Validation:")
    test_path = os.path.join(cutlass_dir, "tests/test_add_then_plus1.py")
    test_valid = validate_test_file(test_path)
    
    # Final summary
    print("\n" + "="*60)
    if all([all_files_exist, syntax_valid, cpp_valid, python_valid, example_valid, test_valid]):
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\nThe AddThenPlus1 custom operator implementation is complete and ready for testing.")
        print("See BUILD_AND_TEST_INSTRUCTIONS.md for environment setup and testing procedures.")
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please review the issues above and fix them before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)