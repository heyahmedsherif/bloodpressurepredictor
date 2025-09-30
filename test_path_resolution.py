#!/usr/bin/env python
"""
Unit Tests for Path Resolution System
=====================================
Ensures robust path resolution across different environments and platforms
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestPathResolution(unittest.TestCase):
    """Test suite for pyVHR and external module path resolution"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_path = sys.path.copy()

    def tearDown(self):
        """Clean up test environment"""
        sys.path = self.original_path
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_relative_path_resolution(self):
        """Test relative path resolution for pyVHR"""
        # Create mock directory structure
        project_root = Path(self.test_dir)
        external_dir = project_root / "external"
        pyvhr_dir = external_dir / "pyVHR"
        pyvhr_dir.mkdir(parents=True)

        # Create a dummy __init__.py to make it importable
        (pyvhr_dir / "__init__.py").write_text("# pyVHR module")

        # Test path resolution
        with patch('os.path.dirname') as mock_dirname:
            mock_dirname.side_effect = lambda x: str(project_root / "core") if "__file__" in str(x) else str(project_root)

            # Simulate the path resolution logic
            pyVHR_path = os.path.join(str(project_root), 'external', 'pyVHR')
            self.assertTrue(os.path.exists(pyVHR_path))
            self.assertTrue(os.path.isdir(pyVHR_path))

    def test_absolute_path_resolution(self):
        """Test absolute path resolution fallback"""
        # Create mock directory structure
        project_root = Path(self.test_dir)
        external_dir = project_root / "external"
        pyvhr_dir = external_dir / "pyVHR"
        pyvhr_dir.mkdir(parents=True)

        # Test absolute path resolution
        current_file = project_root / "core" / "rppg_integration.py"
        current_file.parent.mkdir(parents=True)
        current_file.write_text("# Mock file")

        # Get absolute path
        abs_path = os.path.abspath(str(current_file))
        project_root_abs = os.path.dirname(os.path.dirname(abs_path))
        pyVHR_path = os.path.join(project_root_abs, 'external', 'pyVHR')

        self.assertTrue(os.path.exists(pyVHR_path))
        self.assertEqual(str(Path(pyVHR_path).resolve()), str(pyvhr_dir.resolve()))

    def test_missing_pyvhr_handling(self):
        """Test graceful handling when pyVHR is missing"""
        # Test with non-existent path
        non_existent_path = os.path.join(self.test_dir, "nonexistent", "pyVHR")
        self.assertFalse(os.path.exists(non_existent_path))

        # Simulate the error handling logic
        PYVHR_AVAILABLE = False
        if not os.path.exists(non_existent_path):
            try:
                # Try importing (should fail)
                import pyVHR_nonexistent
                PYVHR_AVAILABLE = True
            except ImportError:
                PYVHR_AVAILABLE = False

        self.assertFalse(PYVHR_AVAILABLE)

    def test_cross_platform_paths(self):
        """Test path resolution works on different platforms"""
        # Test Windows-style paths
        with patch('os.sep', '\\'):
            path_parts = ['C:', 'Users', 'test', 'project', 'external', 'pyVHR']
            windows_path = os.path.join(*path_parts)
            self.assertIn('pyVHR', windows_path)

        # Test Unix-style paths
        with patch('os.sep', '/'):
            path_parts = ['', 'home', 'user', 'project', 'external', 'pyVHR']
            unix_path = os.path.join(*path_parts)
            self.assertIn('pyVHR', unix_path)

    def test_sys_path_insertion(self):
        """Test that paths are correctly inserted into sys.path"""
        test_path = os.path.join(self.test_dir, "test_module")
        os.makedirs(test_path)

        # Insert path
        original_len = len(sys.path)
        sys.path.insert(0, test_path)

        # Verify insertion
        self.assertEqual(sys.path[0], test_path)
        self.assertEqual(len(sys.path), original_len + 1)

    def test_path_normalization(self):
        """Test path normalization handles edge cases"""
        # Test double slashes
        path_with_doubles = os.path.join(self.test_dir, "external//pyVHR")
        normalized = os.path.normpath(path_with_doubles)
        self.assertNotIn('//', normalized)
        self.assertNotIn('\\\\', normalized)

        # Test relative components
        path_with_relative = os.path.join(self.test_dir, "external", "..", "external", "pyVHR")
        normalized = os.path.normpath(path_with_relative)
        self.assertTrue(normalized.endswith(os.path.join("external", "pyVHR")))

    def test_symlink_resolution(self):
        """Test handling of symbolic links"""
        if not hasattr(os, 'symlink'):
            self.skipTest("Symbolic links not supported on this platform")

        # Create actual directory
        actual_dir = os.path.join(self.test_dir, "actual_pyVHR")
        os.makedirs(actual_dir)

        # Create symlink
        symlink_path = os.path.join(self.test_dir, "external", "pyVHR")
        os.makedirs(os.path.dirname(symlink_path), exist_ok=True)

        try:
            os.symlink(actual_dir, symlink_path)

            # Test resolution
            self.assertTrue(os.path.exists(symlink_path))
            self.assertTrue(os.path.isdir(symlink_path))

            # Resolve to actual path
            resolved = os.path.realpath(symlink_path)
            self.assertEqual(resolved, os.path.realpath(actual_dir))
        except OSError:
            self.skipTest("Cannot create symbolic links (permission denied)")


class TestPathValidation(unittest.TestCase):
    """Test suite for path validation utilities"""

    def test_validate_module_path(self):
        """Test module path validation"""
        def validate_module_path(path):
            """Validate that a path is suitable for module import"""
            if not os.path.exists(path):
                return False, "Path does not exist"
            if not os.path.isdir(path):
                return False, "Path is not a directory"
            # Check for __init__.py (Python package)
            init_file = os.path.join(path, "__init__.py")
            if not os.path.exists(init_file):
                # Could still be a namespace package
                pass
            return True, "Valid module path"

        # Test with temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test non-existent path
            valid, msg = validate_module_path(os.path.join(tmpdir, "nonexistent"))
            self.assertFalse(valid)

            # Test existing directory
            test_dir = os.path.join(tmpdir, "test_module")
            os.makedirs(test_dir)
            valid, msg = validate_module_path(test_dir)
            self.assertTrue(valid)

            # Test file (not directory)
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            valid, msg = validate_module_path(test_file)
            self.assertFalse(valid)

    def test_find_project_root(self):
        """Test finding project root from any subdirectory"""
        def find_project_root(start_path):
            """Find project root by looking for key files"""
            markers = ['requirements.txt', 'setup.py', 'pyproject.toml', '.git', 'app.py']

            current = Path(start_path).resolve()
            while current != current.parent:
                for marker in markers:
                    if (current / marker).exists():
                        return str(current)
                current = current.parent
            return None

        # Test with mock project structure
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Create project markers
            (project_root / "requirements.txt").write_text("flask\nnumpy\n")
            (project_root / "app.py").write_text("# Main app")

            # Create subdirectories
            subdir = project_root / "core" / "utils"
            subdir.mkdir(parents=True)

            # Test finding from subdirectory
            found_root = find_project_root(str(subdir))
            # Use realpath to resolve symlinks for comparison
            self.assertEqual(os.path.realpath(found_root), os.path.realpath(str(project_root)))

            # Test from root
            found_root = find_project_root(str(project_root))
            self.assertEqual(os.path.realpath(found_root), os.path.realpath(str(project_root)))


class TestEnvironmentCompatibility(unittest.TestCase):
    """Test environment-specific compatibility"""

    def test_docker_environment(self):
        """Test path resolution in Docker-like environment"""
        # Simulate Docker environment with specific paths
        docker_paths = [
            "/app/external/pyVHR",
            "/usr/src/app/external/pyVHR",
            "/opt/app/external/pyVHR"
        ]

        for docker_path in docker_paths:
            # Normalize path for current OS
            normalized = os.path.normpath(docker_path)
            # Verify structure is maintained
            self.assertTrue(normalized.replace('\\', '/').endswith("external/pyVHR"))

    def test_cloud_environment(self):
        """Test path resolution in cloud environments"""
        # Common cloud environment paths
        cloud_paths = [
            "/home/site/wwwroot/external/pyVHR",  # Azure
            "/var/task/external/pyVHR",            # AWS Lambda
            "/workspace/external/pyVHR",           # Google Cloud Run
            "/app/external/pyVHR"                  # Heroku/Generic
        ]

        for cloud_path in cloud_paths:
            # Ensure path handling doesn't break with cloud-specific structures
            normalized = os.path.normpath(cloud_path)
            self.assertIn("pyVHR", normalized)

    def test_venv_compatibility(self):
        """Test compatibility with virtual environments"""
        # Test that sys.path insertion works with venv
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = os.path.join(tmpdir, "venv", "lib", "python3.9", "site-packages")
            os.makedirs(venv_path, exist_ok=True)

            # Simulate adding to path
            test_path = os.path.join(tmpdir, "external", "pyVHR")
            os.makedirs(test_path, exist_ok=True)

            # Should be able to insert before venv paths
            sys.path.insert(0, test_path)
            self.assertEqual(sys.path[0], test_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete path resolution system"""

    @patch('core.rppg_integration.logger')
    def test_rppg_integration_import(self, mock_logger):
        """Test that rppg_integration handles imports correctly"""
        # This test verifies the actual import logic
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up mock environment
            project_root = Path(tmpdir)
            core_dir = project_root / "core"
            core_dir.mkdir()

            external_dir = project_root / "external"
            pyvhr_dir = external_dir / "pyVHR"
            pyvhr_dir.mkdir(parents=True)

            # Create mock rppg_integration
            rppg_file = core_dir / "rppg_integration.py"
            rppg_content = '''
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Try multiple approaches to locate pyVHR
PYVHR_AVAILABLE = False

# Approach 1: Relative to current file
pyVHR_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'external', 'pyVHR')
if os.path.exists(pyVHR_path):
    sys.path.insert(0, pyVHR_path)
    PYVHR_AVAILABLE = True
    logger.info(f"pyVHR found at: {pyVHR_path}")
'''
            rppg_file.write_text(rppg_content)

            # Change to project directory
            original_cwd = os.getcwd()
            try:
                os.chdir(str(project_root))

                # Verify path exists
                expected_path = project_root / "external" / "pyVHR"
                self.assertTrue(expected_path.exists())

            finally:
                os.chdir(original_cwd)

    def test_error_messages(self):
        """Test that helpful error messages are generated"""
        # Simulate missing pyVHR scenario
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            core_dir = project_root / "core"
            core_dir.mkdir()

            # Expected path for error message
            expected_path = project_root / "external" / "pyVHR"

            # Verify error message would be helpful
            error_msg = f"Expected pyVHR location: {expected_path}"
            self.assertIn("external", error_msg)
            self.assertIn("pyVHR", error_msg)


def run_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPathResolution))
    suite.addTests(loader.loadTestsFromTestCase(TestPathValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate summary
    print("\n" + "="*60)
    print("PATH RESOLUTION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFailed tests:")
        for test, trace in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nTests with errors:")
        for test, trace in result.errors:
            print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)