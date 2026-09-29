"""The process-wide overrides must work across worker threads."""

from concurrent.futures import ThreadPoolExecutor

from cuda_sft.llm import get_llm_client
from cuda_sft.runtime import deps


def test_install_reset_and_nested_use() -> None:
    first = deps.Deps(sleep_fn=lambda _: None)
    second = deps.Deps(llm_factory=lambda role: role)
    with deps.use(first):
        assert deps.current() is first
        with deps.use(second):
            assert deps.current() is second
        assert deps.current() is first
    assert deps.current() == deps.Deps()


def test_injected_llm_factory_receives_role() -> None:
    with deps.use(deps.Deps(llm_factory=lambda role: f"fake:{role}")):
        assert get_llm_client(role="critic") == "fake:critic"


def test_install_visible_to_another_thread() -> None:
    active = deps.Deps(sleep_fn=lambda _: None)
    with deps.use(active), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(deps.current).result() is active
