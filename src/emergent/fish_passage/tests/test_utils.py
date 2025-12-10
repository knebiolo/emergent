import numpy as np
import logging

from emergent.fish_passage import utils


def test_safe_build_kdtree_none():
    tree = utils.safe_build_kdtree(None)
    assert tree is None


def test_safe_build_kdtree_empty():
    tree = utils.safe_build_kdtree(np.empty((0, 2)))
    assert tree is None


def test_safe_build_kdtree_valid():
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    tree = utils.safe_build_kdtree(pts)
    assert tree is not None
    # query a point
    d, idx = tree.query([0.1, 0.1])
    assert idx == 0


def test_safe_log_exception_fallback(capfd):
    # Force logger.exception to raise by setting logger to a dummy that raises
    class BadLogger:
        def exception(self, *args, **kwargs):
            raise RuntimeError('logger failed')

    old_logger = utils.logger
    try:
        utils.logger = BadLogger()
        try:
            raise ValueError('boom')
        except Exception as e:
            utils.safe_log_exception('test message', e, key='value')
        captured = capfd.readouterr()
        # should have printed LOGGING FAILURE to stderr
        assert 'LOGGING FAILURE' in captured.err
    finally:
        utils.logger = old_logger
