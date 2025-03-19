# Create a snapshot of your current environment
pip freeze > current_environment.txt

# Check versions of key packages
python -c "import pkg_resources; print('numpy:', pkg_resources.get_distribution('numpy').version)"
python -c "import pkg_resources; print('tokenizers:', pkg_resources.get_distribution('tokenizers').version)"
python -c "import pkg_resources; print('transformers:', pkg_resources.get_distribution('transformers').version)"
python -c "import pkg_resources; print('huggingface_hub:', pkg_resources.get_distribution('huggingface_hub').version)"

"""
Analyze dependency conflicts for new packages without installing them.
"""
import sys
import subprocess
import json
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

def get_installed_packages():
    """Get currently installed packages and versions."""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def get_package_dependencies(package_name, version=None):
    """Get dependencies for a package without installing it."""
    package_spec = f"{package_name}=={version}" if version else package_name
    try:
        # Use pip to get package metadata in JSON format
        cmd = [sys.executable, "-m", "pip", "install", package_spec, "--dry-run", "--report", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error getting dependencies for {package_spec}:")
            print(result.stderr)
            return []
        
        report = json.loads(result.stdout)
        
        # Extract dependencies from the report
        if 'install' in report and len(report['install']) > 0:
            for pkg in report['install']:
                if pkg['metadata']['name'].lower() == package_name.lower():
                    return pkg['metadata'].get('requires_dist', [])
        
        return []
    except Exception as e:
        print(f"Error analyzing {package_spec}: {e}")
        return []

def check_compatibility(package_name, version=None):
    """Check if a package is compatible with currently installed packages."""
    package_spec = f"{package_name}=={version}" if version else package_name
    installed = get_installed_packages()
    
    # Get dependencies for the package
    dependencies = get_package_dependencies(package_name, version)
    
    conflicts = []
    for dep in dependencies:
        # Parse dependency string
        parts = dep.split(';')
        req = parts[0].strip()
        
        # Skip optional dependencies
        if len(parts) > 1 and 'extra ==' in parts[1]:
            continue
            
        try:
            pkg_resources.require(req)
        except VersionConflict as e:
            conflicts.append(f"Conflict: {e}")
        except DistributionNotFound as e:
            conflicts.append(f"Missing: {e}")
        except Exception as e:
            conflicts.append(f"Error: {e}")
    
    if conflicts:
        print(f"\n{package_spec} has conflicts:")
        for conflict in conflicts:
            print(f"  - {conflict}")
        return False
    else:
        print(f"{package_spec} appears compatible with your environment.")
        return True

def main():
    # List of packages we want to add for PDF knowledge base
    packages_to_check = [
        ("sentence-transformers", "2.2.2"),
        ("faiss-cpu", "1.7.4"),
        ("pdfplumber", "0.7.6")
    ]
    
    print("Analyzing potential dependency conflicts...\n")
    
    # Check each package
    all_compatible = True
    for package, version in packages_to_check:
        if not check_compatibility(package, version):
            all_compatible = False
    
    # Check for specific issues with huggingface_hub
    try:
        import huggingface_hub
        if hasattr(huggingface_hub, 'cached_download'):
            print("\nhuggingface_hub has cached_download function. ✓")
        else:
            print("\nWARNING: huggingface_hub is missing cached_download function. This will cause issues with sentence-transformers. ✗")
            all_compatible = False
    except ImportError:
        print("\nhuggingface_hub is not installed.")
    
    # Final report
    if all_compatible:
        print("\n✅ All packages appear compatible with your environment.")
    else:
        print("\n❌ Conflicts detected. Please review the issues above.")

if __name__ == "__main__":
    main()
