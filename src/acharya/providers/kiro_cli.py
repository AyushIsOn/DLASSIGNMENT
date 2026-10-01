"""Sandboxed Kiro CLI provider with bounded, tools-disabled execution."""

from __future__ import annotations

import os
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from acharya.providers.base import (
    ProviderConfigurationError,
    ProviderOutputError,
    ProviderRequest,
    ProviderResult,
    ProviderTimeoutError,
    ProviderTransportError,
)

KIRO_CLI_VERSION = "2.26.0"
_MAX_STREAM_BYTES = 1_048_576
_ALLOWED_ENVIRONMENT = (
    "HOME",
    "PATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "KIRO_API_KEY",
)


class KiroCLIProvider:
    name = "kiro-cli"

    def __init__(
        self,
        executable: Path | str | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        resolved = Path(executable or shutil.which("kiro-cli") or "").expanduser().resolve()
        if not resolved.is_file():
            raise ProviderConfigurationError("kiro-cli is unavailable")
        mode = resolved.stat().st_mode
        if not stat.S_ISREG(mode) or mode & 0o022 or not mode & stat.S_IXUSR:
            raise ProviderConfigurationError("kiro-cli permissions are unsafe")
        self.executable = resolved
        source = {key: value for key, value in os.environ.items() if key in _ALLOWED_ENVIRONMENT}
        if environment is not None:
            source.update(
                {
                    key: str(value)
                    for key, value in environment.items()
                    if key in _ALLOWED_ENVIRONMENT
                }
            )
        self.environment = {
            key: str(source[key])
            for key in _ALLOWED_ENVIRONMENT
            if source.get(key) is not None and str(source[key])
        }
        self.version = self._read_version()

    def _read_version(self) -> str:
        try:
            result = subprocess.run(
                [str(self.executable), "--version"],
                shell=False,
                cwd="/",
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise ProviderConfigurationError("kiro-cli version check failed") from error
        actual_version = result.stdout.decode("utf-8", "replace").strip()
        if result.returncode != 0 or actual_version != f"kiro-cli {KIRO_CLI_VERSION}":
            raise ProviderConfigurationError("unsupported kiro-cli version")
        return KIRO_CLI_VERSION

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()

    @classmethod
    def _bounded_output(
        cls, process: subprocess.Popen[bytes], timeout: float
    ) -> tuple[int, bytes, bytes]:
        selector = selectors.DefaultSelector()
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        streams: dict[int, bytearray] = {}
        for stream, target in (
            (process.stdout, stdout_buffer),
            (process.stderr, stderr_buffer),
        ):
            if stream is not None:
                descriptor = stream.fileno()
                os.set_blocking(descriptor, False)
                streams[descriptor] = target
                selector.register(descriptor, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                cls._terminate(process)
                process.wait()
                raise ProviderTimeoutError()
            for key, _ in selector.select(min(remaining, 0.1)):
                descriptor = key.fd
                block = os.read(descriptor, 65_536)
                if not block:
                    selector.unregister(descriptor)
                    continue
                target = streams[descriptor]
                if len(target) < _MAX_STREAM_BYTES:
                    target.extend(block[: _MAX_STREAM_BYTES - len(target)])
        try:
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as error:
            cls._terminate(process)
            process.wait()
            raise ProviderTimeoutError() from error
        return code, bytes(stdout_buffer), bytes(stderr_buffer)

    def generate(self, request: ProviderRequest) -> ProviderResult:
        timeout = request.deadline_monotonic - time.monotonic()
        if timeout <= 0:
            raise ProviderTimeoutError()
        with tempfile.TemporaryDirectory(prefix="acharya-kiro-") as directory:
            isolated = Path(directory)
            isolated.chmod(0o700)
            command = [
                str(self.executable),
                "chat",
                "--no-interactive",
                "--trust-tools=",
                "--wrap=never",
                request.prompt.text,
            ]
            try:
                process = subprocess.Popen(
                    command,
                    shell=False,
                    cwd=isolated,
                    env=self.environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                code, stdout, _stderr = self._bounded_output(process, timeout)
            except ProviderTimeoutError:
                raise
            except OSError as error:
                raise ProviderTransportError() from error
        if code != 0:
            raise ProviderTransportError(exit_code=code)
        text = stdout.decode("utf-8", "replace").strip()
        if not text or len(stdout) >= _MAX_STREAM_BYTES:
            raise ProviderOutputError()
        return ProviderResult(text=text, used_chunk_ids=(), provider=self.name)
