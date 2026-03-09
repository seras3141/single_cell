import numpy as np
import pandas as pd
import pytest
from src.postprocessing.cell_tracking import CellTracker3D, TrackingConfig

def test_extract_cell_properties():
    mask = np.zeros((20, 20), dtype=int)
    mask[2:8, 2:8] = 1
    mask[10:18, 10:18] = 2
    tracker = CellTracker3D(TrackingConfig(min_area=5, max_area=100))
    props = tracker.extract_cell_properties(mask)
    assert len(props) == 2
    assert set(['x', 'y', 'label', 'area']).issubset(props.columns)

def test_extract_3d_centers():
    stack = np.zeros((3, 20, 20), dtype=int)
    stack[0, 2:8, 2:8] = 1
    stack[1, 10:18, 10:18] = 2
    tracker = CellTracker3D()
    centers = tracker.extract_3d_centers(stack)
    assert len(centers) == 3
    for z, df in centers:
        assert isinstance(df, pd.DataFrame)

def test_track_cells():
    stack = np.zeros((3, 20, 20), dtype=int)
    stack[0, 2:8, 2:8] = 1
    stack[1, 2:8, 2:8] = 1
    stack[2, 2:8, 2:8] = 1
    tracker = CellTracker3D()
    tracked = tracker.track_cells(stack)
    assert tracked.shape == stack.shape
    assert tracked.dtype == np.int32
    assert np.max(tracked) > 0
