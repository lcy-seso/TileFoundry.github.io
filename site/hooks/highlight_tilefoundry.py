"""Colour the `tilefoundry` command and its subcommand inside code blocks.

Pygments emits a command name as untagged text, so no token class exists for
CSS to select. This wraps each occurrence inside a <pre> in a span the
stylesheet can reach. Prose and inline <code> are left alone.
"""

import re

COMMAND = "tilefoundry"
SUBCOMMANDS = ("analyze", "check", "schedule", "spec", "tutorial")
CMD_CLASS = "tf-cmd"
SUB_CLASS = "tf-subcmd"

_PRE = re.compile(r"<pre\b.*?</pre>", re.S)
_TAG = re.compile(r"<[^>]+>")
# Not part of a longer identifier, and not the host in tilefoundry.github.io.
_WORD = re.compile(rf"\b{COMMAND}\b(?![./\w-])")
_CMD_SPAN = f'<span class="{CMD_CLASS}">{COMMAND}</span>'
# The subcommand follows the command across Pygments' whitespace markup, so the
# match has to span tags; anchoring on the span above keeps it unambiguous.
_SUBCOMMAND = re.compile(
    re.escape(_CMD_SPAN) + rf"((?:<[^>]*>|\s)*)\b({'|'.join(SUBCOMMANDS)})\b"
)


def _wrap_command(pre: str) -> str:
    out, last = [], 0
    for tag in _TAG.finditer(pre):                 # rewrite only between tags
        out.append(_WORD.sub(_CMD_SPAN, pre[last:tag.start()]))
        out.append(tag.group(0))
        last = tag.end()
    out.append(_WORD.sub(_CMD_SPAN, pre[last:]))
    return "".join(out)


def _wrap_subcommand(pre: str) -> str:
    return _SUBCOMMAND.sub(
        lambda m: f'{_CMD_SPAN}{m.group(1)}<span class="{SUB_CLASS}">{m.group(2)}</span>',
        pre,
    )


def on_post_page(output, page, config):
    return _PRE.sub(lambda m: _wrap_subcommand(_wrap_command(m.group(0))), output)
