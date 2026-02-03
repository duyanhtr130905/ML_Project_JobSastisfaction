"""
Lightweight structure verification script
Verifies the translation system structure without requiring heavy dependencies
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False


def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        files = list(Path(dirpath).rglob("*.*"))
        print(f"✓ {description}: {dirpath} ({len(files)} files)")
        return True
    else:
        print(f"✗ {description}: {dirpath} NOT FOUND")
        return False


def verify_structure():
    """Verify the translation system structure"""
    print("=" * 60)
    print("Translation System - Structure Verification")
    print("=" * 60)
    print()
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    checks = []
    
    # Core files
    print("Core Translation Engine:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "core", "translator.py"),
        "MarianMT Translator"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "core", "__init__.py"),
        "Core __init__"
    ))
    print()
    
    # RAG files
    print("RAG Translation Memory:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "rag", "translation_memory.py"),
        "Translation Memory"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "rag", "__init__.py"),
        "RAG __init__"
    ))
    print()
    
    # Knowledge Graph files
    print("Knowledge Graph:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "knowledge_graph", "terminology.py"),
        "Terminology Manager"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "knowledge_graph", "__init__.py"),
        "KG __init__"
    ))
    print()
    
    # API files
    print("API Backend:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "api", "main.py"),
        "FastAPI Backend"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "api", "__init__.py"),
        "API __init__"
    ))
    print()
    
    # Configuration
    print("Configuration:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "config", "settings.py"),
        "Settings"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "requirements.txt"),
        "Requirements"
    ))
    print()
    
    # Docker
    print("Docker Configuration:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "Dockerfile"),
        "Dockerfile"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "docker-compose.yml"),
        "Docker Compose"
    ))
    print()
    
    # Documentation
    print("Documentation:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "README.md"),
        "Main README"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "examples", "usage_examples.py"),
        "Usage Examples"
    ))
    print()
    
    # Drupal
    print("Drupal Integration:")
    checks.append(check_directory_exists(
        os.path.join(base_dir, "drupal", "translation_module"),
        "Drupal Module"
    ))
    print()
    
    # Mobile
    print("Mobile Application:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "mobile", "package.json"),
        "Mobile Package"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "mobile", "src", "App.tsx"),
        "Mobile App"
    ))
    print()
    
    # Analytics
    print("Analytics:")
    checks.append(check_file_exists(
        os.path.join(base_dir, "analytics", "init.sql"),
        "Analytics SQL"
    ))
    checks.append(check_file_exists(
        os.path.join(base_dir, "analytics", "README.md"),
        "Analytics README"
    ))
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    print(f"Structure Verification: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All structure checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the API: cd api && python main.py")
        print("3. Or use Docker: docker-compose up -d")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed")
        return 1


def verify_python_syntax():
    """Verify Python files have valid syntax"""
    print("\n" + "=" * 60)
    print("Python Syntax Verification")
    print("=" * 60)
    print()
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    python_files = [
        "core/translator.py",
        "rag/translation_memory.py",
        "knowledge_graph/terminology.py",
        "api/main.py",
        "config/settings.py",
        "examples/usage_examples.py"
    ]
    
    all_valid = True
    for py_file in python_files:
        filepath = os.path.join(base_dir, py_file)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    compile(f.read(), filepath, 'exec')
                print(f"✓ {py_file}: Valid syntax")
            except SyntaxError as e:
                print(f"✗ {py_file}: Syntax error - {e}")
                all_valid = False
        else:
            print(f"✗ {py_file}: File not found")
            all_valid = False
    
    if all_valid:
        print("\n✓ All Python files have valid syntax!")
    else:
        print("\n✗ Some Python files have syntax errors")
    
    return all_valid


def main():
    """Run all verifications"""
    structure_ok = verify_structure()
    syntax_ok = verify_python_syntax()
    
    if structure_ok == 0 and syntax_ok:
        print("\n" + "=" * 60)
        print("✓ Translation System Successfully Verified!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ Verification Failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
