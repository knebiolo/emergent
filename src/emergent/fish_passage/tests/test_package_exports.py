def test_import_exports():
    import emergent.fish_passage as fp
    assert hasattr(fp, 'geo_to_pixel')
    assert hasattr(fp, 'geo_to_pixel_from_inv')
    assert hasattr(fp, 'compute_affine_from_hecras')
