import os
import sys
from pathlib import Path


def redirect_output(path: Path) -> None:
    """Appends this process's stdout and stderr to path from now on."""
    sys.stdout.flush()
    sys.stderr.flush()
    # replacing the file descriptors, not sys.stdout, also catches what C extensions,
    # torch.compile and DataLoader workers write
    file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(file_descriptor, sys.stdout.fileno())
    os.dup2(file_descriptor, sys.stderr.fileno())
    os.close(file_descriptor)
    # a file makes Python buffer whole blocks, so a crash could lose the last lines
    sys.stdout.reconfigure(line_buffering=True)
