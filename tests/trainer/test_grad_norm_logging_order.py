from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())


def _load_method(file_path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    module = ast.parse(file_path.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
            break
    raise AssertionError(f"Could not find {class_name}.{method_name} in {file_path}")


def _collect_strategy_call_lines(target_method: ast.FunctionDef, attr_name: str) -> list[int]:
    call_lines = []
    for child in ast.walk(target_method):
        if not isinstance(child, ast.Call):
            continue
        if not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr != attr_name:
            continue
        owner = child.func.value
        if not isinstance(owner, ast.Attribute):
            continue
        if owner.attr != "strategy":
            continue
        if not isinstance(owner.value, ast.Name) or owner.value.id != "self":
            continue
        call_lines.append(child.lineno)
    return call_lines


@pytest.mark.unit
@pytest.mark.parametrize(
    ("relative_path", "class_name", "method_name"),
    [
        ("openrlhf/trainer/sft_trainer.py", "SFTTrainer", "fit"),
        ("openrlhf/trainer/dpo_trainer.py", "DPOTrainer", "fit"),
        ("openrlhf/trainer/rm_trainer.py", "RewardModelTrainer", "fit"),
        ("openrlhf/trainer/ray/ppo_actor.py", "ActorPPOTrainer", "training_step"),
        ("openrlhf/trainer/ray/ppo_critic.py", "CriticPPOTrainer", "training_step"),
    ],
)
def test_trainers_consume_grad_norm_from_optimizer_step(
    relative_path: str,
    class_name: str,
    method_name: str,
) -> None:
    file_path = PROJECT_ROOT / relative_path
    target_method = _load_method(file_path, class_name, method_name)
    optimizer_step_lines = _collect_strategy_call_lines(target_method, "optimizer_step")
    grad_norm_lines = _collect_strategy_call_lines(target_method, "get_grad_norm")
    source = file_path.read_text()

    assert optimizer_step_lines, f"Expected an optimizer_step call in {class_name}.{method_name}"
    assert grad_norm_lines, f"Expected a get_grad_norm call in {class_name}.{method_name}"
    assert min(grad_norm_lines) < max(optimizer_step_lines)
    assert "grad_norm=" in source
