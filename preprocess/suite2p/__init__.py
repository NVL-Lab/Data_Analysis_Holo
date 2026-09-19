"""Suite2p processing helpers and pipeline package."""

__all__ = [
    "get_settings",
    "obtain_bad_frames_from_voltage_rec",
    "prepare_ops_1st_pass",
    "process_1_session_suite2p_offline",
    "process_single_session",
]


def __getattr__(name):
    if name in __all__:
        try:
            from suite2p.utils.core import (
                get_settings,
                obtain_bad_frames_from_voltage_rec,
                prepare_ops_1st_pass,
                process_1_session_suite2p_offline,
                process_single_session,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc

        return {
            "get_settings": get_settings,
            "obtain_bad_frames_from_voltage_rec": obtain_bad_frames_from_voltage_rec,
            "prepare_ops_1st_pass": prepare_ops_1st_pass,
            "process_1_session_suite2p_offline": process_1_session_suite2p_offline,
            "process_single_session": process_single_session,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
