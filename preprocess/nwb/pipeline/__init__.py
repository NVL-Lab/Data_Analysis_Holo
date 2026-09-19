"""NWB conversion pipeline entry points."""

__all__ = ["convert_all_experiments_to_nwb", "convert_bruker_images_to_nwb", "main"]


def __getattr__(name):
    if name == "convert_all_experiments_to_nwb":
        try:
            from nwb.convert_to_nwb import convert_all_experiments_to_nwb
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        return convert_all_experiments_to_nwb
    if name == "convert_bruker_images_to_nwb":
        try:
            from nwb.convert_to_nwb import convert_bruker_images_to_nwb
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        return convert_bruker_images_to_nwb
    if name == "main":
        try:
            from nwb.convert_to_nwb import main
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
