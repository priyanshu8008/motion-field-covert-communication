import sys
import os

RAFT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../RAFT")
)

if RAFT_ROOT not in sys.path:
    sys.path.insert(0, RAFT_ROOT)
