"""The installed distribution and source must identify the same release."""
import importlib.metadata
import unittest
import mlirbench


class PackageVersion(unittest.TestCase):
    def test_installed_version_matches_source(self):
        self.assertEqual(mlirbench.__version__, "0.3.0rc5")
        self.assertEqual(importlib.metadata.version("mlirbench"), mlirbench.__version__)
