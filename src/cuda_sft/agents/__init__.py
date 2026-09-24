"""LLM agent roles used by the kernel and knowledge LangGraphs.

``KernelDialectAgent`` / ``KnowledgeAgent`` remain registries in their
own packages. This package holds the actual model-calling roles
(generator, later repairer / critic) and their typed contracts.
"""

from cuda_sft.agents.contracts import CriticResult, GenerateResult
from cuda_sft.agents.critic import KernelCritic, should_run_critic
from cuda_sft.agents.difficulty import plan_topology
from cuda_sft.agents.generate import (
    assistant_state_update,
    cancel_speculative,
    complete_chat,
    enqueue_speculative_repair,
)
from cuda_sft.agents.repairer import (
    classify_compile_error,
    classify_refval_error,
    repair_budget,
    repair_budget_for_difficulty,
    repair_system_prompt,
    wrap_repair_user,
)

__all__ = [
    "CriticResult",
    "GenerateResult",
    "KernelCritic",
    "plan_topology",
    "should_run_critic",
    "assistant_state_update",
    "cancel_speculative",
    "classify_compile_error",
    "classify_refval_error",
    "repair_budget",
    "repair_budget_for_difficulty",
    "complete_chat",
    "enqueue_speculative_repair",
    "repair_system_prompt",
    "wrap_repair_user",
]
