"""How the harness talks to an agent. `python -m agent.harness run <dir> --responder module:Class` constructs
module.Class(run_dir=..., agent="claude" | "openai", should_stop=<callable>, log=<callable>); the object needs

    ask(run, turn, prompt, thread) -> (answer_text, meta)   thread is "new" (a fresh conversation) or "resume"; meta is a
                                                            dict with at least "ok" and "refused", plus token usage
    finished(run)                                           (optional) called once the run has its final answer

and raises Stopped while it waits if the run's STOP file appears. The runs in the paper used a responder that passed each
turn to a separate machine where the agent CLI answered (`claude -p` for Claude Opus 5.5, `codex exec` for Codex with
GPT-6.1 Sol, both at their highest reasoning effort, with the agents' own tools switched off and the one-line system prompt
in assets/agent_system_prompt.md); agent/dryrun/fake_agent.py is a scripted responder for dry runs.
"""


class Stopped(Exception):
    """the run's STOP file appeared while waiting for an answer"""
