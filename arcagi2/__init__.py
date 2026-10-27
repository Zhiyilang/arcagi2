"""arcagi2 - a deterministic program-synthesis solver for ARC-AGI-2.

Pure standard library. A closed transform DSL is searched for programs that
reproduce every demonstration pair exactly; nothing is trained, nothing is
guessed at inference time, and every consistency check is exact equality.
"""

from . import grids
from . import objects
from . import dsl
from . import infer
from . import solver
from . import taskio

__version__ = "0.1.0"

__all__ = ["grids", "objects", "dsl", "infer", "solver", "taskio"]
