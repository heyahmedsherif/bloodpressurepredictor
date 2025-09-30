#!/usr/bin/env python
"""
Path Resolution Utilities
=========================
Robust utilities for resolving external module paths across different environments
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class PathResolver:
    """Handles path resolution for external modules with multiple fallback strategies"""

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize PathResolver

        Args:
            project_root: Optional project root path. If not provided, will be auto-detected.
        """
        self.project_root = project_root or self._find_project_root()
        self._validated_paths = {}  # Cache for validated paths

    @staticmethod
    def _find_project_root(start_path: Optional[str] = None) -> str:
        """
        Find project root by looking for marker files

        Args:
            start_path: Starting path for search (defaults to current file location)

        Returns:
            Project root path or current directory if not found
        """
        markers = ['requirements.txt', 'setup.py', 'pyproject.toml', '.git', 'app.py', 'README.md']

        if start_path is None:
            start_path = os.path.abspath(os.path.dirname(__file__))

        current = Path(start_path).resolve()

        # Search up the directory tree
        while current != current.parent:
            for marker in markers:
                if (current / marker).exists():
                    logger.debug(f"Project root found at: {current}")
                    return str(current)
            current = current.parent

        # Fallback to current directory
        fallback = os.getcwd()
        logger.warning(f"Could not find project root markers, using: {fallback}")
        return fallback

    def resolve_module_path(
        self,
        module_name: str,
        relative_path: str = "external",
        fallback_strategies: bool = True
    ) -> Tuple[Optional[str], bool]:
        """
        Resolve path to an external module using multiple strategies

        Args:
            module_name: Name of the module to find (e.g., 'pyVHR')
            relative_path: Relative path from project root (default: 'external')
            fallback_strategies: Whether to try multiple resolution strategies

        Returns:
            Tuple of (resolved_path, is_available)
        """
        # Check cache first
        cache_key = f"{relative_path}/{module_name}"
        if cache_key in self._validated_paths:
            return self._validated_paths[cache_key]

        strategies = [
            self._try_relative_to_project,
            self._try_absolute_path,
            self._try_environment_variable,
            self._try_python_path,
            self._try_site_packages,
        ]

        if not fallback_strategies:
            strategies = [strategies[0]]  # Only use primary strategy

        for strategy in strategies:
            path, available = strategy(module_name, relative_path)
            if available:
                logger.info(f"Module '{module_name}' found using {strategy.__name__}: {path}")
                self._validated_paths[cache_key] = (path, available)
                return path, available

        logger.warning(f"Module '{module_name}' not found after trying all strategies")
        self._validated_paths[cache_key] = (None, False)
        return None, False

    def _try_relative_to_project(self, module_name: str, relative_path: str) -> Tuple[Optional[str], bool]:
        """Try resolving relative to project root"""
        if not self.project_root:
            return None, False

        module_path = os.path.join(self.project_root, relative_path, module_name)
        if os.path.exists(module_path) and os.path.isdir(module_path):
            return os.path.abspath(module_path), True
        return None, False

    def _try_absolute_path(self, module_name: str, relative_path: str) -> Tuple[Optional[str], bool]:
        """Try using absolute path resolution"""
        try:
            # Get absolute path of current file's parent
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            module_path = os.path.join(project_root, relative_path, module_name)

            if os.path.exists(module_path) and os.path.isdir(module_path):
                return os.path.abspath(module_path), True
        except Exception as e:
            logger.debug(f"Absolute path resolution failed: {e}")
        return None, False

    def _try_environment_variable(self, module_name: str, relative_path: str) -> Tuple[Optional[str], bool]:
        """Try using environment variable for module path"""
        env_var = f"{module_name.upper()}_PATH"
        module_path = os.environ.get(env_var)

        if module_path and os.path.exists(module_path) and os.path.isdir(module_path):
            logger.info(f"Using {env_var} environment variable: {module_path}")
            return os.path.abspath(module_path), True
        return None, False

    def _try_python_path(self, module_name: str, relative_path: str) -> Tuple[Optional[str], bool]:
        """Try importing module if it's already in Python path"""
        try:
            module = __import__(module_name)
            if hasattr(module, '__file__'):
                module_path = os.path.dirname(os.path.abspath(module.__file__))
                return module_path, True
        except ImportError:
            pass
        return None, False

    def _try_site_packages(self, module_name: str, relative_path: str) -> Tuple[Optional[str], bool]:
        """Try finding in site-packages"""
        import site

        for site_dir in site.getsitepackages():
            module_path = os.path.join(site_dir, module_name)
            if os.path.exists(module_path) and os.path.isdir(module_path):
                return os.path.abspath(module_path), True
        return None, False

    def add_to_path(self, module_path: str, prepend: bool = True) -> bool:
        """
        Add module path to sys.path

        Args:
            module_path: Path to add
            prepend: Whether to prepend (True) or append (False) to sys.path

        Returns:
            True if successfully added, False otherwise
        """
        if not os.path.exists(module_path):
            logger.error(f"Cannot add non-existent path to sys.path: {module_path}")
            return False

        if module_path in sys.path:
            logger.debug(f"Path already in sys.path: {module_path}")
            return True

        if prepend:
            sys.path.insert(0, module_path)
        else:
            sys.path.append(module_path)

        logger.debug(f"Added to sys.path: {module_path}")
        return True

    def validate_module(self, module_path: str) -> Tuple[bool, str]:
        """
        Validate that a path is suitable for module import

        Args:
            module_path: Path to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if not os.path.exists(module_path):
            return False, f"Path does not exist: {module_path}"

        if not os.path.isdir(module_path):
            return False, f"Path is not a directory: {module_path}"

        # Check for __init__.py (traditional package)
        init_file = os.path.join(module_path, "__init__.py")
        if os.path.exists(init_file):
            return True, f"Valid Python package with __init__.py: {module_path}"

        # Could be a namespace package (PEP 420)
        # Check if it contains any .py files
        py_files = list(Path(module_path).glob("*.py"))
        if py_files:
            return True, f"Valid namespace package with Python files: {module_path}"

        return False, f"Directory exists but doesn't appear to be a Python module: {module_path}"

    def get_diagnostic_info(self) -> dict:
        """
        Get diagnostic information about path resolution

        Returns:
            Dictionary with diagnostic information
        """
        info = {
            "project_root": self.project_root,
            "python_executable": sys.executable,
            "python_version": sys.version,
            "sys_path": sys.path[:5],  # First 5 entries
            "cwd": os.getcwd(),
            "validated_paths": self._validated_paths,
            "platform": sys.platform,
        }

        # Check for common environment variables
        env_vars = ["PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"]
        info["environment"] = {var: os.environ.get(var, "Not set") for var in env_vars}

        return info


def resolve_and_add_module(
    module_name: str,
    relative_path: str = "external",
    required: bool = False
) -> bool:
    """
    Convenience function to resolve and add a module to sys.path

    Args:
        module_name: Name of the module (e.g., 'pyVHR')
        relative_path: Relative path from project root
        required: Whether to raise an error if module not found

    Returns:
        True if module was found and added, False otherwise

    Raises:
        ImportError: If module is required but not found
    """
    resolver = PathResolver()
    module_path, available = resolver.resolve_module_path(module_name, relative_path)

    if available and module_path:
        success = resolver.add_to_path(module_path)
        if success:
            logger.info(f"Successfully added {module_name} to path")
            return True

    if required:
        # Provide helpful error message
        diagnostic_info = resolver.get_diagnostic_info()
        error_msg = f"""
        Required module '{module_name}' not found!

        Expected locations checked:
        1. {os.path.join(resolver.project_root, relative_path, module_name)}
        2. Environment variable: {module_name.upper()}_PATH
        3. Python path (already installed)
        4. Site-packages

        Current project root: {diagnostic_info['project_root']}
        Python executable: {diagnostic_info['python_executable']}

        To fix this issue:
        1. Ensure {module_name} exists in the 'external' directory
        2. Or install it: pip install {module_name}
        3. Or set environment variable: export {module_name.upper()}_PATH=/path/to/{module_name}
        """
        raise ImportError(error_msg)

    logger.warning(f"Optional module '{module_name}' not found, continuing without it")
    return False


def ensure_external_modules() -> dict:
    """
    Ensure all known external modules are available

    Returns:
        Dictionary with module availability status
    """
    modules = {
        "pyVHR": {"path": "external", "required": False},
        "webcam-pulse-detector": {"path": "external", "required": False},
    }

    status = {}
    for module_name, config in modules.items():
        try:
            available = resolve_and_add_module(
                module_name,
                config["path"],
                config["required"]
            )
            status[module_name] = available
        except ImportError as e:
            logger.error(f"Failed to resolve {module_name}: {e}")
            status[module_name] = False

    return status


if __name__ == "__main__":
    # Self-test when run directly
    logging.basicConfig(level=logging.DEBUG)

    print("Path Resolution Utilities - Self Test")
    print("=" * 50)

    resolver = PathResolver()
    print(f"\nProject root: {resolver.project_root}")

    print("\nTesting pyVHR resolution:")
    path, available = resolver.resolve_module_path("pyVHR")
    print(f"  Path: {path}")
    print(f"  Available: {available}")

    print("\nDiagnostic information:")
    info = resolver.get_diagnostic_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, list):
            print(f"  {key}: {value[:3]}...")  # Show first 3 items
        else:
            print(f"  {key}: {value}")

    print("\nEnsuring all external modules:")
    status = ensure_external_modules()
    for module, available in status.items():
        status_str = "✓ Available" if available else "✗ Not found"
        print(f"  {module}: {status_str}")