# NWB conversion pipeline

This package mirrors the Suite2p layout:

- `preprocess/nwb/pipeline/` contains the executable conversion workflow
- `preprocess/nwb/utils/` holds helper utilities
- `preprocess/nwb/legacy/` holds older or compatibility code

The canonical submission entry point is the package-based SLURM array runner:

```bash
python -m preprocess.nwb.pipeline.submit --help
```

Compatibility wrappers are also available at the package root for local tooling:

```bash
python -m preprocess.nwb.submit --help
python -m preprocess.nwb.run_session --help
```

The conversion stack is intentionally lazy-loaded: importing the package does not require the full NWB runtime to be installed, and the environment-specific conversion logic only loads when a conversion task is executed.
