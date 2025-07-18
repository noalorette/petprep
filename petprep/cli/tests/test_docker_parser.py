import importlib
import sys
from pathlib import Path
import subprocess

# Add wrapper package to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'wrapper' / 'src'))


def test_docker_get_parser():
    module = importlib.import_module('petprep_docker.__main__')
    parser = module.get_parser()
    assert parser is not None


def test_docker_main_help(monkeypatch, capsys):
    module = importlib.import_module('petprep_docker.__main__')
    monkeypatch.setattr(module, 'check_docker', lambda: 1)
    monkeypatch.setattr(module, 'check_image', lambda img: True)
    monkeypatch.setattr(module, 'check_memory', lambda img: 16000)

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        # handle docker version query
        if '--format' in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=b'20.10')
        return subprocess.CompletedProcess(cmd, 0)

    def fake_check_output(cmd):
        captured['cmd'] = cmd
        return b'usage: petprep bids_dir output_dir {participant}\n\noptional arguments:'

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    monkeypatch.setattr(module.subprocess, 'check_output', fake_check_output)
    monkeypatch.setattr(module, 'merge_help', lambda a, b: 'merged')

    sys.argv = ['petprep-docker', '--help']
    ret = module.main()
    assert ret == 0
    assert captured['cmd'][:3] == ['docker', 'run', '--rm']
    assert '-h' in captured['cmd']
    assert '-i' in captured['cmd']
    assert module.__name__


def test_docker_main_version(monkeypatch):
    module = importlib.import_module('petprep_docker.__main__')
    monkeypatch.setattr(module, 'check_docker', lambda: 1)
    monkeypatch.setattr(module, 'check_image', lambda img: True)
    monkeypatch.setattr(module, 'check_memory', lambda img: 16000)

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        if '--format' in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=b'20.10')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, 'run', fake_run)

    sys.argv = ['petprep-docker', '--version']
    ret = module.main()
    assert ret == 0
    cmd = calls[-1]
    assert cmd[:3] == ['docker', 'run', '--rm']
    assert '--version' in cmd


def test_docker_command_options(monkeypatch, tmp_path):
    module = importlib.import_module('petprep_docker.__main__')
    monkeypatch.setattr(module, 'check_docker', lambda: 1)
    monkeypatch.setattr(module, 'check_image', lambda img: True)
    monkeypatch.setattr(module, 'check_memory', lambda img: 16000)

    bids_dir = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids_dir.mkdir()
    out_dir.mkdir()
    work_dir.mkdir()

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        if '--format' in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=b'20.10')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, 'run', fake_run)

    sys.argv = [
        'petprep-docker',
        str(bids_dir),
        str(out_dir),
        'participant',
        '--work-dir',
        str(work_dir),
        '--output-spaces',
        'MNI152Lin',
    ]
    ret = module.main()
    assert ret == 0
    cmd = calls[-1]
    joined = ' '.join(cmd)
    assert f'{work_dir}:/scratch' in joined
    assert '--output-spaces' in cmd
    assert 'MNI152Lin' in cmd
    