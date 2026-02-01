from emergent.salmon_abm.summary import summary as Summary


def test_find_h5_files(tmp_path):
    root_h5 = tmp_path / "root.h5"
    root_h5.write_bytes(b"")
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child_h5 = child_dir / "child.h5"
    child_h5.write_bytes(b"")

    s = Summary(str(tmp_path), str(tmp_path / "dummy.tif"))
    assert set(s.h5_files) == {str(root_h5), str(child_h5)}
