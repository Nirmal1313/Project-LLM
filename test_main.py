"""
Unit tests for main.py
"""

import unittest
from main import greet


class TestMain(unittest.TestCase):
    """Test cases for main module functions."""

    def test_greet(self):
        """Test the greet function."""
        self.assertEqual(greet("Alice"), "Hello, Alice!")
        self.assertEqual(greet("World"), "Hello, World!")

if __name__ == "__main__":
    unittest.main()
