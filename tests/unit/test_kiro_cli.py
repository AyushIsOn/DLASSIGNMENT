from __future__ import annotations

import stat
import time
from pathlib import Path

from acharya.providers.base import ProviderRequest
from acharya.providers.kiro_cli import KiroCLIProvider
from acharya.rag.prompt import RenderedPrompt


def test_kiro_cli_exact_version_tools_disabled_and_isolated(tmp_path: Path) -> None:
    executable = tmp_path / "kiro-cli"
    executable.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then echo 'kiro-cli 2.26.0'; "
        "else echo 'KIRO_API_OK'; fi\n",
        encoding="utf-8",
    )
    executable.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )
    provider = KiroCLIProvider(executable)
    prompt = RenderedPrompt("KIRO_API_OK", b"KIRO_API_OK", 11, 8, ())
    result = provider.generate(ProviderRequest(prompt, time.monotonic() + 5))
    assert result.text == "KIRO_API_OK"
    assert provider.version == "2.26.0"
