"""The pieces of the inference source that are a temporal downscaler's own.

A downscaler reads the states at the two bounds `t` and `t + T` of a window and predicts the states between them in
one model call. The evaluation scores it with an init time as the left bound `t`, the lead time as the offset inside
the window and the run's lead time as the window `T`: one init time is one window and one model call.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping

import numpy as np
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta


def trajectory_trained(metadata: Mapping) -> bool:
    """Whether the checkpoint was trained on forecast trajectories: a dataloader dataset of the stored training config
    has a `trajectory` block (anemoi-training's `TrajectoryDataset`), directly or in its `dataset_config`."""
    dataloader = (metadata.get("config") or {}).get("dataloader")
    for stage in dataloader.values() if isinstance(dataloader, Mapping) else ():
        datasets = stage.get("datasets") if isinstance(stage, Mapping) else None
        for entry in datasets.values() if isinstance(datasets, Mapping) else ():
            for block in (entry, entry.get("dataset_config") if isinstance(entry, Mapping) else None):
                if isinstance(block, Mapping) and block.get("trajectory") is not None:
                    return True
    return False


def downscaler_window(checkpoint: object) -> tuple[datetime.timedelta, list[datetime.timedelta]]:
    """The window `T` between the two input bounds and the offsets of the outputs inside it, from the checkpoint's
    explicit input and target times and its data frequency.

    anemoi-inference labels output `s` with the offset `(s + 1) * T / n` (`n` the number of data steps in the window)
    whatever the target times are, so targets that are not `1, 2, ...` steps after the left bound would be scored at
    the wrong valid times: they are refused."""
    inputs = [int(index) for index in checkpoint.input_explicit_times]
    targets = [int(index) for index in checkpoint.target_explicit_times]
    if len(inputs) != 2:
        raise ValueError(f"a temporal downscaler needs exactly two input times (the bounds), got {inputs}")
    frequency = frequency_to_timedelta(checkpoint.data_frequency)
    window = (inputs[1] - inputs[0]) * frequency
    leads = [(index - inputs[0]) * frequency for index in targets]
    labels = [(s + 1) * frequency for s in range(len(targets))]
    if not leads or leads != labels:
        raise ValueError(
            f"the temporal downscaler's target times {targets} (input times {inputs}) are not supported yet: "
            f"anemoi-inference labels the outputs {', '.join(frequency_to_string(t) for t in labels)} after the left "
            "bound whatever the target times are (a left-boundary output, for example), see "
            "MeteoSwiss/anemoi-evaluation#5"
        )
    return window, leads


def window_input_state(
    runner: object, dataset: str, start: datetime.datetime, window: datetime.timedelta, constants: dict
) -> dict:
    """Input state of `dataset` for the window `[start, start + window]`, dated `start`.

    A private coupling to anemoi-inference (NFR-17): `TemporalDownscalerMultiOutRunner.execute()` builds this state
    inline and exposes no method for it, while its `run()` takes the state ready made. As there, the prognostics and
    dynamic forcings are read over the two bounds with `ref_date_index=0`, and the constant forcings are read once
    (here at the first window, kept in `constants` by dataset), repeated for the two bounds and reused for every
    window; the three are combined with `Runner._combine_states`. Needs anemoi-inference 0.13 or later (`dates=`).
    To be replaced by the runner's own method once it has one."""
    if dataset not in constants:
        state = runner.constant_forcings_inputs[dataset].create_input_state(
            dates=[start], constant=True, ref_date_index=0
        )
        fields = {key: np.concatenate([value, value], axis=0) for key, value in state["fields"].items()}
        constants[dataset] = {**state, "fields": fields}
    dates = [start, start + window]
    prognostics = runner.prognostics_inputs[dataset].create_input_state(
        dates=dates, select_reference_date=True, ref_date_index=0
    )
    forcings = runner.dynamic_forcings_inputs[dataset].create_input_state(
        dates=dates, select_reference_date=True, ref_date_index=0
    )
    return runner._combine_states(prognostics, {**constants[dataset], "date": start}, forcings)
