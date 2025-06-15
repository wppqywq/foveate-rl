#!/usr/bin/env python3
"""
Project Structure Setup Script
Organizes files into proper GitHub-ready structure and fixes import paths.
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the recommended project directory structure."""
    directories = [
        'experiments',
        'tests', 
        'docs',
        'results',
        'data',
        'logs'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")

def move_files_to_experiments():
    """Move experiment files to experiments directory."""
    files_to_move = [
        'train_baseline.py',
        'setup_phase1.py', 
        'debug_opencv.py',
        'quick_test.py'
    ]
    
    for file_name in files_to_move:
        if os.path.exists(file_name):
            destination = f'experiments/{file_name}'
            if not os.path.exists(destination):
                shutil.move(file_name, destination)
                print(f"✅ Moved {file_name} → experiments/{file_name}")
            else:
                print(f"⚠️  File already exists: experiments/{file_name}")
        else:
            print(f"ℹ️  File not found: {file_name} (may already be moved)")

def create_empty_test_files():
    """Create empty test files for the testing framework."""
    test_files = [
        'tests/__init__.py',
        'tests/test_transforms.py',
        'tests/test_models.py', 
        'tests/test_metrics.py',
        'tests/test_integration.py'
    ]
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            Path(test_file).touch()
            print(f"✅ Created test file: {test_file}")

def create_documentation_files():
    """Create placeholder documentation files."""
    doc_files = {
        'docs/architecture.md': '# Technical Architecture\n\nTODO: Document the technical architecture',
        'docs/experiments.md': '# Experimental Protocols\n\nTODO: Document experimental procedures',
        'docs/api_reference.md': '# API Reference\n\nTODO: Auto-generated API documentation'
    }
    
    for doc_file, content in doc_files.items():
        if not os.path.exists(doc_file):
            with open(doc_file, 'w') as f:
                f.write(content)
            print(f"✅ Created documentation: {doc_file}")

def verify_imports():
    """Verify that the import structure is working."""
    print("\n🔍 Verifying import structure...")
    
    try:
        # Test imports from experiments directory
        test_script = """
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

try:
    from fovea_lib import FovealTransform, create_baseline_model
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
"""
        
        # Write test script
        with open('experiments/test_imports.py', 'w') as f:
            f.write(test_script)
        
        # Run test
        import subprocess
        result = subprocess.run([sys.executable, 'experiments/test_imports.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Import verification successful!")
        else:
            print(f"❌ Import verification failed: {result.stderr}")
        
        # Clean up test file
        os.remove('experiments/test_imports.py')
        
    except Exception as e:
        print(f"⚠️  Could not verify imports: {e}")

def check_current_structure():
    """Check and display current project structure."""
    print("\n📁 Current project structure:")
    
    def print_tree(path, prefix="", max_depth=2, current_depth=0):
        if current_depth >= max_depth:
            return
            
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item}")
            
            if os.path.isdir(item_path) and current_depth < max_depth - 1:
                extension = "    " if is_last else "│   "
                print_tree(item_path, prefix + extension, max_depth, current_depth + 1)
    
    print_tree(".")

def main():
    """Main setup function."""
    print("🚀 Setting up Foveal Vision Project Structure")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('fovea_lib'):
        print("❌ Error: fovea_lib directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    print("📁 Creating directory structure...")
    create_directory_structure()
    
    print("\n📦 Moving files to experiments/...")
    move_files_to_experiments()
    
    print("\n🧪 Creating test framework...")
    create_empty_test_files()
    
    print("\n📚 Creating documentation structure...")
    create_documentation_files()
    
    print("\n🔍 Verifying import structure...")
    verify_imports()
    
    print("\n📋 Final project structure:")
    check_current_structure()
    
    print("\n" + "=" * 50)
    print("🎉 Project structure setup complete!")
    print("\n📋 Next steps:")
    print("1. Run: python experiments/setup_phase1.py")
    print("2. Run: python experiments/train_baseline.py --epochs 30")
    print("3. Initialize git: git init && git add . && git commit -m 'Initial commit'")
    print("4. Push to GitHub!")

if __name__ == "__main__":
    main()