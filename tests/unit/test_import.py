"""Smoke checks for modules required by the pipeline."""


def test_pipeline_modules_import() -> None:
    import cuda_sft.graph
    import cuda_sft.knowledge.graph
    import cuda_sft.refval.runner
    import cuda_sft.store

    assert hasattr(cuda_sft.graph.build_graph(), "invoke")
