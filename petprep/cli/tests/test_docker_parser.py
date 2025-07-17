import importlib
import sys
from pathlib import Path

# Add wrapper package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'wrapper' / 'src'))


def test_docker_get_parser():
    module = importlib.import_module('petprep_docker.__main__')
    parser = module.get_parser()
    assert parser is not None