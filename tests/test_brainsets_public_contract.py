def test_required_public_dataset_classes_are_exported():
    from torch_brain import datasets

    for name in (
        "Neuroprobe2025",
        "NeuroprobeV2",
        "KelesBYD2024",
        "BerezutskayaPippi2022",
    ):
        assert getattr(datasets, name).__name__ == name


def test_exact_public_pipeline_ids_and_assets_are_packaged():
    from torch_brain.pipeline._cli.utils import PIPELINES_PATH, get_available_brainsets

    available = set(get_available_brainsets())
    assert {"neuroprobe_2025", "keles_byd_2024", "berezutskaya_pippi_2022"} <= available
    for pipeline_id in ("keles_byd_2024", "berezutskaya_pippi_2022"):
        pipeline = PIPELINES_PATH / pipeline_id
        assert len(list((pipeline / "labels").glob("*.csv"))) == 30
        assert (pipeline / "brain_areas" / "brain_area_labels.csv").is_file()
