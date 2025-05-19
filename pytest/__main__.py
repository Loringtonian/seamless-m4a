import sys
import unittest
from pathlib import Path


def main() -> int:
    loader = unittest.defaultTestLoader
    suite = loader.discover(Path.cwd())
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
