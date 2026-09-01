import compare_segmentations


def test_version():
    assert isinstance(compare_segmentations.__version__, str)
