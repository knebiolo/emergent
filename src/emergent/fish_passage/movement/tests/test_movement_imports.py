def test_package_importable():
    from emergent.fish_passage import movement
    assert hasattr(movement, 'swim_core')
    assert hasattr(movement, 'drag_and_battery')
    assert hasattr(movement, 'calc_battery')
    assert hasattr(movement, 'merged_battery')
    assert hasattr(movement, 'bout_distance')
    assert hasattr(movement, 'time_to_fatigue')
