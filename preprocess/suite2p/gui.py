"""Simple desktop interface for submitting Suite2p SLURM jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import re


def _day_index(value: str) -> str:
    if not re.fullmatch(r"D.*", value):
        raise argparse.ArgumentTypeError(
            "Day index must start with 'D' (for example, D10)."
        )
    return value


class Suite2pSubmitter(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Suite2p Pipeline")
        self.minsize(700, 500)

        self.dataframe = tk.StringVar()
        self.raw_root = tk.StringVar()
        self.output_root = tk.StringVar()
        self.frame_rate = tk.StringVar(value="29.752")
        self.executor = tk.StringVar(value="slurm")
        self.conda_env = tk.StringVar(
            value=os.environ.get("SUITE2P_CONDA_ENV", "")
        )
        self.session_date = tk.StringVar()
        self.day_index = tk.StringVar()
        self.mouse_id = tk.StringVar()
        self.max_parallel = tk.StringVar(value="1")
        self._build_form()

    def _build_form(self) -> None:
        form = ttk.Frame(self, padding=16)
        form.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        form.columnconfigure(1, weight=1)

        fields = (
            ("Dataframe (.parquet)", self.dataframe, self._choose_dataframe),
            ("Raw data directory", self.raw_root, self._choose_raw_root),
            ("Output directory", self.output_root, self._choose_output_root),
        )
        for row, (label, variable, command) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(form, textvariable=variable).grid(
                row=row, column=1, sticky="ew", padx=8, pady=5
            )
            ttk.Button(form, text="Browse...", command=command).grid(
                row=row, column=2, pady=5
            )

        executor_row = len(fields)
        ttk.Label(form, text="Executor").grid(
            row=executor_row, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            form,
            textvariable=self.executor,
            values=("slurm", "local"),
            state="readonly",
        ).grid(row=executor_row, column=1, sticky="ew", padx=8, pady=5)

        options = (
            ("Frame rate", self.frame_rate),
            ("Conda environment", self.conda_env),
            ("Session date (optional)", self.session_date),
            ("Day index (optional)", self.day_index),
            ("Mouse ID (optional)", self.mouse_id),
            ("Maximum parallel jobs", self.max_parallel),
        )
        row_index = len(fields) + 1
        for label, variable in options:
            ttk.Label(form, text=label).grid(
                row=row_index, column=0, sticky="w", pady=5
            )
            ttk.Entry(form, textvariable=variable).grid(
                row=row_index, column=1, sticky="ew", padx=8, pady=5
            )
            if label == "Day index (optional)":
                ttk.Label(
                    form,
                    text="Day index must start with D, for example D10.",
                    wraplength=350,
                ).grid(
                    row=row_index + 1,
                    column=1,
                    sticky="w",
                    padx=8,
                    pady=(0, 4),
                )
                row_index += 2
            else:
                row_index += 1

        self.submit_button = ttk.Button(
            form, text="Submit Suite2p job", command=self._submit
        )
        self.submit_button.grid(
            row=row_index + 1,
            column=0,
            columnspan=3,
            pady=(14, 8),
        )

        self.output = tk.Text(form, height=10, wrap="word", state="disabled")
        self.output.grid(
            row=row_index + 2,
            column=0,
            columnspan=3,
            sticky="nsew",
        )
        form.rowconfigure(row_index + 2, weight=1)

    def _choose_dataframe(self) -> None:
        path = filedialog.askopenfilename(
            title="Select dataframe",
            filetypes=(("Parquet files", "*.parquet"), ("All files", "*.*")),
        )
        if path:
            self.dataframe.set(path)

    def _choose_raw_root(self) -> None:
        path = filedialog.askdirectory(title="Select raw data directory")
        if path:
            self.raw_root.set(path)

    def _choose_output_root(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_root.set(path)

    def _write_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)
        self.output.configure(state="disabled")

    def _submit(self) -> None:
        try:
            dataframe = Path(self.dataframe.get())
            raw_root = Path(self.raw_root.get())
            output_root = Path(self.output_root.get())
            if not dataframe.is_file():
                raise ValueError("Select an existing dataframe.")
            if not raw_root.is_dir():
                raise ValueError("Select an existing raw data directory.")
            float(self.frame_rate.get())
            conda_env = self.conda_env.get().strip()
            if self.executor.get() == "slurm" and not conda_env:
                raise ValueError(
                    "Enter the conda environment used by the SLURM job."
                )
            max_parallel = int(self.max_parallel.get())
            if max_parallel < 1:
                raise ValueError("Maximum parallel jobs must be at least 1.")
            day_index = self.day_index.get().strip()
            if day_index:
                _day_index(day_index)
        except (ValueError, argparse.ArgumentTypeError) as error:
            messagebox.showerror("Invalid settings", str(error))
            return

        command = [
            sys.executable,
            "-m",
            "preprocess.suite2p.pipeline.submit",
            "--dataframe",
            str(dataframe),
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(output_root),
            "--frame-rate",
            self.frame_rate.get(),
            "--executor",
            self.executor.get(),
            "--max-parallel",
            str(max_parallel),
        ]
        if self.executor.get() == "slurm":
            command.extend(("--conda-env", conda_env))
        optional = (
            ("--session-date", self.session_date.get().strip()),
            ("--day-index", day_index),
            ("--mouse-id", self.mouse_id.get().strip()),
        )
        for option, value in optional:
            if value:
                command.extend((option, value))

        self.submit_button.configure(state="disabled")
        self._write_output("Submitting job...\n")
        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            details = getattr(error, "stderr", None) or str(error)
            self._write_output(details)
            messagebox.showerror("Submission failed", details)
        else:
            self._write_output(result.stdout or "Job submitted.")
            messagebox.showinfo("Submitted", result.stdout or "Job submitted.")
        finally:
            self.submit_button.configure(state="normal")


def main() -> None:
    Suite2pSubmitter().mainloop()


if __name__ == "__main__":
    main()
