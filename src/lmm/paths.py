"""Where the user's files live — never where the package lives.

Before this module each runtime computed its root as "two directories above
my own source file". That silently meant the repository root while the
package sat at the repository root, and it broke the moment the package moved
under `src/` — the field trial caught it: every engine call failed and every
question came back "I don't know" in zero milliseconds, because `.env` and
the model directory were being looked for inside `src/`. Installed from PyPI
it would have been worse: the search would land in `site-packages`, where a
user's model can never be.

A library must resolve user data against the USER's location, not its own:
`LMM_HOME` if set, otherwise the working directory. `.env` is searched from
the working directory upwards, the way tools like dotenv and git behave, so
running from a subdirectory still finds the project's file.
"""
import os


def home():
    """Directory holding the user's `models/`, memories and `.env`."""
    return os.environ.get("LMM_HOME") or os.getcwd()


def find_env(levels=4):
    """Nearest `.env` from the working directory upwards; None if there is
    none. Explicit `LMM_ENV` wins."""
    named = os.environ.get("LMM_ENV")
    if named:
        return named if os.path.exists(named) else None
    here = os.path.abspath(home())
    for _ in range(levels):
        candidate = os.path.join(here, ".env")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def under(*parts):
    """A path under the user's home, e.g. under('models', 'qwen-3b')."""
    return os.path.join(home(), *parts)
