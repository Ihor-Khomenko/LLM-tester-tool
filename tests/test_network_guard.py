import socket

import pytest


def test_network_is_blocked_by_default():
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.socket()
