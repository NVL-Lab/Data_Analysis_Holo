"""Pipeline entry points for Suite2p job submission and execution."""

__all__ = [
    "main",
    "read_manifest_entry",
    "run_session",
    "select_rows",
    "write_manifest",
]


def __getattr__(name):
    if name in {"read_manifest_entry", "select_rows", "write_manifest"}:
        try:
            from .manifest import read_manifest_entry, select_rows, write_manifest
        except (ImportError, ModuleNotFoundError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc

        return {
            "read_manifest_entry": read_manifest_entry,
            "select_rows": select_rows,
            "write_manifest": write_manifest,
        }[name]
    if name == "run_session":
        try:
            from .run_session import run_session
        except (ImportError, ModuleNotFoundError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc

        return run_session
    if name == "main":
        try:
            from .submit import main
        except (ImportError, ModuleNotFoundError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
