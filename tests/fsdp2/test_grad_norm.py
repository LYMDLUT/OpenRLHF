from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_flash_attn_stubs_if_missing() -> None:
    if importlib.util.find_spec("flash_attn") is not None:
        return

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__path__ = []  # type: ignore[attr-defined]
    flash_attn_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)

    bert_padding_module = types.ModuleType("flash_attn.bert_padding")
    bert_padding_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding_module.index_first_axis = lambda *args, **kwargs: None
    bert_padding_module.pad_input = lambda *args, **kwargs: None
    bert_padding_module.rearrange = lambda *args, **kwargs: None
    bert_padding_module.unpad_input = lambda *args, **kwargs: (None, None, None, None, None)

    utils_module = types.ModuleType("flash_attn.utils")
    utils_module.__path__ = []  # type: ignore[attr-defined]
    utils_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.utils", loader=None, is_package=True)

    distributed_module = types.ModuleType("flash_attn.utils.distributed")
    distributed_module.__spec__ = importlib.machinery.ModuleSpec("flash_attn.utils.distributed", loader=None)
    distributed_module.all_gather = lambda *args, **kwargs: None

    sys.modules["flash_attn"] = flash_attn_module
    sys.modules["flash_attn.bert_padding"] = bert_padding_module
    sys.modules["flash_attn.utils"] = utils_module
    sys.modules["flash_attn.utils.distributed"] = distributed_module


_install_flash_attn_stubs_if_missing()

from openrlhf.utils.fsdp2 import strategy as fsdp2_strategy_module
from openrlhf.utils.fsdp2.utils import get_grad_norm_dtensor

FSDP2Strategy = fsdp2_strategy_module.FSDP2Strategy


def _build_strategy_args() -> SimpleNamespace:
    return SimpleNamespace(
        fsdp2_cp_size=1,
        fsdp2_tp_size=1,
        fsdp2_tp_loss_parallel=False,
        param_dtype="fp32",
        fsdp2_cpu_offload=False,
        fsdp2_reshard_after_forward=True,
        fsdp2_enable_sleep=False,
    )


@pytest.mark.unit
def test_get_grad_norm_dtensor_returns_zero_without_gradients() -> None:
    model = nn.Linear(2, 1, bias=False)

    assert get_grad_norm_dtensor(model) == 0.0


@pytest.mark.unit
def test_get_grad_norm_dtensor_matches_regular_tensor_norm() -> None:
    model = nn.Linear(2, 1, bias=False)
    loss = model(torch.tensor([[3.0, 4.0]])).sum()

    loss.backward()

    assert get_grad_norm_dtensor(model) == pytest.approx(5.0)


@pytest.mark.unit
def test_strategy_get_grad_norm_reads_live_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = FSDP2Strategy(args=_build_strategy_args())
    model = nn.Linear(2, 1, bias=False)
    expected_norm = 1.23
    captured_models = []

    def _fake_get_grad_norm_dtensor(unwrapped_model):
        captured_models.append(unwrapped_model)
        return expected_norm

    monkeypatch.setattr(fsdp2_strategy_module, "get_grad_norm_dtensor", _fake_get_grad_norm_dtensor)

    assert strategy.get_grad_norm(model) == expected_norm
    assert captured_models == [model]


@pytest.mark.unit
def test_optimizer_step_returns_false_when_micro_step_does_not_update() -> None:
    strategy = FSDP2Strategy(args=_build_strategy_args())
    strategy.accumulated_gradient = 2
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = model(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()

    assert strategy.optimizer_step(optimizer, model, scheduler=None) is False


@pytest.mark.unit
def test_optimizer_step_uses_supplied_grad_norm_for_clipping(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = FSDP2Strategy(max_norm=1.0, args=_build_strategy_args())
    strategy.accumulated_gradient = 1
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    provided_grad_norm = 7.5
    clip_calls = []

    def _fake_clip_grad_norm_dtensor(unwrapped_model, max_norm, norm_type=2.0, total_norm=None):
        clip_calls.append(
            {
                "model": unwrapped_model,
                "max_norm": max_norm,
                "norm_type": norm_type,
                "total_norm": total_norm,
            }
        )
        return provided_grad_norm

    monkeypatch.setattr(fsdp2_strategy_module, "clip_grad_norm_dtensor", _fake_clip_grad_norm_dtensor)
    monkeypatch.setattr(
        fsdp2_strategy_module,
        "get_grad_norm_dtensor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should reuse supplied grad norm")),
    )

    loss = model(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()

    assert strategy.optimizer_step(optimizer, model, scheduler=None, grad_norm=provided_grad_norm) is True
    assert clip_calls == [
        {
            "model": model,
            "max_norm": 1.0,
            "norm_type": 2.0,
            "total_norm": provided_grad_norm,
        }
    ]


@pytest.mark.unit
def test_optimizer_step_computes_grad_norm_when_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy = FSDP2Strategy(max_norm=1.0, args=_build_strategy_args())
    strategy.accumulated_gradient = 1
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    expected_norm = 5.5
    call_count = 0

    def _fake_get_grad_norm_dtensor(unwrapped_model):
        nonlocal call_count
        call_count += 1
        return expected_norm

    monkeypatch.setattr(fsdp2_strategy_module, "get_grad_norm_dtensor", _fake_get_grad_norm_dtensor)
    monkeypatch.setattr(
        fsdp2_strategy_module,
        "clip_grad_norm_dtensor",
        lambda *_args, **kwargs: kwargs["total_norm"],
    )

    loss = model(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()

    assert strategy.optimizer_step(optimizer, model, scheduler=None) is True
    assert call_count == 1


@pytest.mark.unit
def test_optimizer_step_zeroes_gradients_after_update() -> None:
    strategy = FSDP2Strategy(args=_build_strategy_args())
    strategy.accumulated_gradient = 1
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss = model(torch.tensor([[1.0, 2.0]])).sum()
    loss.backward()

    strategy.optimizer_step(optimizer, model, scheduler=None)

    assert all(parameter.grad is None for parameter in model.parameters())
