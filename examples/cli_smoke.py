"""Smoke test: build a Hermes agent, run one turn through the StreamParser.

Confirms that:

  1. ``HermesConfig.resolve()`` picks up env / TOML correctly.
  2. ``create_hermes_agent(cfg)`` returns a compiled graph.
  3. The graph's ``stream(...)`` plugs into ``langgraph_stream_parser.StreamParser``
     without any host-specific glue.
  4. ``PrintAdapter`` echoes the agent's response to stdout.

Run:  ``python examples/cli_smoke.py``
Needs: ``ANTHROPIC_API_KEY`` (default model is anthropic:claude-sonnet-4-6).
"""

from __future__ import annotations

from deepagent_hermes import HermesConfig, create_hermes_agent
from langgraph_stream_parser import StreamParser
from langgraph_stream_parser.adapters import PrintAdapter


def main() -> None:
    cfg = HermesConfig.resolve()
    agent = create_hermes_agent(cfg)
    parser = StreamParser()
    adapter = PrintAdapter()
    stream = agent.stream(
        {"messages": [{"role": "user", "content": "Say hello in one word."}]},
        stream_mode="updates",
    )
    for event in parser.parse(stream):
        adapter.handle(event)


if __name__ == "__main__":
    main()
