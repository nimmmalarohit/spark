import os
import sys
from pathlib import Path

def print_project_files(project_root, output_file=None):
    """
    Generate a text file with all Python files in the project,
    including their full paths and content.
    
    Args:
        project_root (str): Path to the project root directory
        output_file (str, optional): Path to save the output. If None, prints to stdout.
    """
    project_path = Path(project_root).resolve()
    
    # Create output file or use stdout
    if output_file:
        f = open(output_file, 'w', encoding='utf-8')
    else:
        f = sys.stdout
    
    try:
        # Find all Python files
        python_files = sorted(project_path.glob('**/*.py'))
        
        # Print summary of files first
        f.write("======= PROJECT FILES SUMMARY =======\n\n")
        for i, file_path in enumerate(python_files, 1):
            rel_path = file_path.relative_to(project_path)
            f.write(f"{i}. {rel_path}\n")
        
        f.write("\n\n======= FILE CONTENTS =======\n\n")
        
        # Print each file with its full path
        for file_path in python_files:
            rel_path = file_path.relative_to(project_path)
            f.write(f"\n\n{'=' * 80}\n")
            f.write(f"FILE: {rel_path}\n")
            f.write(f"FULL PATH: {file_path}\n")
            f.write(f"{'=' * 80}\n\n")
            
            # Read and write file content
            try:
                with open(file_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
                    f.write(content)
            except Exception as e:
                f.write(f"Error reading file: {e}\n")
            
            f.write("\n")
    finally:
        if output_file:
            f.close()
            print(f"Project files printed to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_project.py <project_root> [output_file]")
        sys.exit(1)
    
    project_root = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "project_code.txt"
    
    print_project_files(project_root, output_file)
