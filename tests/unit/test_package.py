import mlirbench


def test_package_version():
    assert mlirbench.__version__ == "0.2.0"


import unittest


class Package(unittest.TestCase):
    def test_version(self):
        test_package_version()
