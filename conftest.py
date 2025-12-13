import sys
from pathlib import Path


def pytest_collection_modifyitems(config, items):
    """Deselect any collected tests under a 'tools' directory when running on Windows.

    This prevents GUI/OpenGL/Qt diagnostic scripts in `tools/` from being
    executed in the same pytest process on Windows (they can cause native
    crashes or platform/plugin initialization failures). Tests under other
    directories are left untouched.
    """
    if not sys.platform.startswith("win"):
        return

    removed = []
    kept = []
    gui_keywords = ('pyqt', 'pyqt5', 'pyqtgraph', 'opengl', 'OpenGL', 'QOpenGL', 'QWidget', 'QApplication')
    for item in items:
        try:
            path_str = str(item.fspath)
            parts = Path(path_str).parts
        except Exception:
            kept.append(item)
            continue

        # Deselect anything under a top-level tools/ directory
        if 'tools' in parts:
            removed.append(item)
            continue

        # Also deselect tests whose source references GUI/OpenGL libraries
        try:
            src = Path(path_str).read_text(errors='ignore')
            lower = src.lower()
            if any(k.lower() in lower for k in gui_keywords):
                removed.append(item)
                continue
        except Exception:
            # If we can't read the file, keep it (safer)
            kept.append(item)
            continue

        kept.append(item)

    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept
        tr = config.pluginmanager.get_plugin('terminalreporter')
        if tr:
            tr.write_sep('-', f'Deselected {len(removed)} tests under tools/ on Windows')
