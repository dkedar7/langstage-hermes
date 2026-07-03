"""Guard: hermes middleware must be async-callable through the AG-UI host.

Since langstage-core 1.0 (ADR 0003) the host renders every turn through the
in-process AG-UI adapter, which drives the graph via ``astream`` — so any
middleware that defines the sync ``wrap_model_call`` MUST also define the async
``awrap_model_call``, or chat crashes on the first message with "Asynchronous
implementation of awrap_model_call is not available" (gh #43).

This structural test catches a new (or edited) middleware that forgets the async
pair — the exact regression that shipped hermes 0.4.0/0.4.1 with a broken chat.
"""
import inspect

import pytest

from langstage_hermes.caching import AnthropicCachingS3Middleware
from langstage_hermes.plugins.event_bus import PluginEventBus
from langstage_hermes.prompts import PromptAssemblyMiddleware
from langstage_hermes.skills.loader import SkillLoaderMiddleware

# Every hermes middleware that can wrap the model call.
_MODEL_CALL_MIDDLEWARE = [
    SkillLoaderMiddleware,
    PromptAssemblyMiddleware,
    AnthropicCachingS3Middleware,
    PluginEventBus,
]


@pytest.mark.parametrize("cls", _MODEL_CALL_MIDDLEWARE, ids=lambda c: c.__name__)
def test_wrap_model_call_has_async_pair(cls):
    """A middleware that defines sync wrap_model_call must also define the async one."""
    if "wrap_model_call" not in cls.__dict__:
        pytest.skip(f"{cls.__name__} does not override wrap_model_call")
    assert "awrap_model_call" in cls.__dict__, (
        f"{cls.__name__} defines wrap_model_call but not awrap_model_call — the AG-UI "
        "host drives the graph via astream and will crash chat on the first message (gh #43)"
    )
    assert inspect.iscoroutinefunction(cls.awrap_model_call), (
        f"{cls.__name__}.awrap_model_call must be `async def`"
    )
