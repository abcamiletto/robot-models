"""Thin model-specific bindings over the shared family implementations."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from robot_models import _catalog as catalog
from robot_models._runtime import JaxRuntime, NumpyRuntime, RuntimeName, TorchRuntime
from robot_models._skinned._model import Gender, SkinnedHand, SkinnedHumanoid
from robot_models._spec import Side
from robot_models.hands._base import Hand
from robot_models.humanoids._base import Humanoid

_RUNTIMES = {"numpy": NumpyRuntime, "torch": TorchRuntime, "jax": JaxRuntime}


def model_for_backend(name: str, backend: RuntimeName, *, module: str) -> type[Any]:
    """Bind one canonical definition to a runtime, following Body Models' import style."""
    definition = catalog.model_spec(name)
    runtime = _RUNTIMES[backend]
    module_base: Any = object
    if backend == "torch":
        from torch import nn

        module_base = nn.Module
    adapter_base = cast(Any, module_base)
    skinned = definition.body_model is not None
    if definition.family == "hands" and skinned:

        class Model(SkinnedHand, adapter_base):
            def __init__(
                self,
                *,
                side: Side = "right",
                shape: Sequence[float] | None = None,
                conform_to: Sequence[float] | None = None,
                model_path: Path | str | None = None,
            ) -> None:
                if backend == "torch":
                    module_base.__init__(self)
                geometry = {"shape": shape, "conform_to": conform_to}
                super().__init__(name, side=side, **geometry, model_path=model_path, runtime=runtime())

    elif definition.family == "hands":

        class Model(Hand, adapter_base):
            def __init__(self, *, side: Side = "right", model_path: Path | str | None = None) -> None:
                if backend == "torch":
                    module_base.__init__(self)
                super().__init__(name, side=side, model_path=model_path, runtime=runtime())

    elif skinned:

        class Model(SkinnedHumanoid, adapter_base):
            def __init__(
                self,
                *,
                gender: Gender = "neutral",
                shape: Sequence[float] | None = None,
                conform_to: Sequence[float] | None = None,
                model_path: Path | str | None = None,
            ) -> None:
                if backend == "torch":
                    module_base.__init__(self)
                geometry = {"shape": shape, "conform_to": conform_to}
                super().__init__(name, gender=gender, **geometry, model_path=model_path, runtime=runtime())

    else:

        class Model(Humanoid, adapter_base):
            def __init__(self, *, model_path: Path | str | None = None) -> None:
                if backend == "torch":
                    module_base.__init__(self)
                super().__init__(name, model_path=model_path, runtime=runtime())

    Model.__name__ = definition.class_name
    Model.__qualname__ = definition.class_name
    Model.__init__.__qualname__ = f"{definition.class_name}.__init__"
    Model.__init__.__module__ = module
    Model.__module__ = module
    Model.__doc__ = f"Canonical {definition.class_name} using {backend} arrays."
    return Model
