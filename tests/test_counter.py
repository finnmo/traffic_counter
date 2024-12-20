# tests/test_counter.py

import pytest
import numpy as np
from traffic_counter.counter import TrafficCounter, CountLine, Direction

def test_count_line_equation():
    line = CountLine(start=(0, 0), end=(10, 10), original_size=(100, 100))
    a, b, c = line.get_line_equation()
    assert a == 10
    assert b == -10
    assert c == 0

def test_process_frame_no_line():
    counter = TrafficCounter()
    with pytest.raises(ValueError):
        counter.process_frame(np.zeros((480, 640, 3)))
