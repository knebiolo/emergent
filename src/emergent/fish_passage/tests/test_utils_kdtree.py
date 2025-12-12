import sys
import io
import numpy as np
from emergent.fish_passage.utils import safe_build_kdtree, safe_log_exception


def test_safe_build_kdtree_none():
    assert safe_build_kdtree(None) is None


def test_safe_build_kdtree_empty():
    assert safe_build_kdtree(np.zeros((0, 2))) is None


def test_safe_build_kdtree_good():
    tree = safe_build_kdtree(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert tree is not None


def test_safe_log_exception_fallback(monkeypatch, capsys):
    # Force logger.exception to raise to exercise fallback
    class E(Exception):
        pass

    def bad_exception(*args, **kwargs):
        raise E('boom')

    import logging
    logger = logging.getLogger('emergent.fish_passage.utils')
    monkeypatch.setattr(logger, 'exception', bad_exception)
    # Call safe_log_exception which should catch and write to stderr
    safe_log_exception('msg', RuntimeError('test'), ctx='x')
    # ensure it didn't raise
