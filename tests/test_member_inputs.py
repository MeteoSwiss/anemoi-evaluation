import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anemoi.evaluation.config import AnemoiInferenceConfig
from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource
from anemoi.evaluation.sources.anemoi_inference import format_init_time

HOUR = datetime.timedelta(hours=1)
T0 = datetime.datetime(2024, 1, 2)
CPU = torch.device("cpu")
N = 5


def test_format_init_time_and_member_inputs_config():
    block = {"dataset": {"cutout": ["/data/m1.zarr", "/boundary/{yyyy}/{mm}/{dd}/bc_{yyyymmdd}_{hh}_m1.zarr"]}, "n": 1}
    assert format_init_time(block, T0 + 12 * HOUR) == {
        "dataset": {"cutout": ["/data/m1.zarr", "/boundary/2024/01/02/bc_20240102_12_m1.zarr"]},
        "n": 1,
    }
    assert format_init_time("{yyyymmddhh}", T0) == "2024010200"
    config = AnemoiInferenceConfig.model_validate({"checkpoint": "x", "members": 2, "member_inputs": [block, block]})
    assert config.member_inputs == [block, block]
    with pytest.raises(ValueError, match="2 input blocks for 3 members"):
        AnemoiInferenceConfig.model_validate({"checkpoint": "x", "members": 3, "member_inputs": [block, block]})


class FakeInput:
    """One member's prognostics at an init time: 10u = 3 (m + 1), 10v = 4 (m + 1), 2t = init hour + m."""

    def __init__(self, member: int, init_time: datetime.datetime) -> None:
        self.member, self.init_time = member, init_time

    def create_input_state(self, *, dates, **kwargs):
        m = self.member
        values = {"10u": 3.0 * (m + 1), "10v": 4.0 * (m + 1), "2t": float(self.init_time.hour + m)}
        fields = {v: np.full((len(dates), N), x, dtype=np.float32) for v, x in values.items()}
        return {"date": dates[-1], "fields": fields}


class EmptyInput:
    def create_input_state(self, *, dates, **kwargs):
        return {"fields": {}}


def combine(*states):
    combined = {"date": states[0]["date"], "fields": {}}
    for state in states:
        combined["fields"].update(state["fields"])
    return combined


def fake_run(input_states, lead_time, return_numpy):
    """Each step adds the step count to the member's initial fields: members stay distinguishable."""
    state = input_states["ds"]
    for k in range(1, int(lead_time / (6 * HOUR)) + 1):
        out = {v: torch.as_tensor(x[-1]) + k for v, x in state["fields"].items()}
        yield {"ds": {"step": 6 * k * HOUR, "date": state["date"] + 6 * k * HOUR, "fields": out}}


def member_inputs_source(members: int = 2) -> InferenceForecastSource:
    source = InferenceForecastSource.__new__(InferenceForecastSource)
    source.dataset_names, source.dataset_name = ["ds"], "ds"
    source.datasets = {"ds": SimpleNamespace(variables=["10u", "10v", "2t"], grid=SimpleNamespace(n=N))}
    source.member_inputs = [{"dataset": f"/m{m + 1}/{{yyyymmdd}}{{hh}}.zarr"} for m in range(members)]
    source.members, source.member_group, source.seed, source.quiet = members, None, 0, False
    source.temporal_downscaling = False
    source.output_offsets, source.output_horizon = [6 * HOUR], 6 * HOUR
    source._initial = {}
    source.runner = SimpleNamespace(
        checkpoint=SimpleNamespace(multi_dataset_metadata={"ds": SimpleNamespace(lagged=[-6 * HOUR, 0 * HOUR])}),
        constant_forcings_inputs={"ds": EmptyInput()},
        dynamic_forcings_inputs={"ds": EmptyInput()},
        _combine_states=combine,
        run=fake_run,
        reference_date=None,
        device=torch.device("cpu"),
        tensor_handlers={"ds": SimpleNamespace(boundary_forcings_providers=[])},
    )
    source._member_input = lambda member, init_time: FakeInput(member, init_time)
    return source


def test_member_inputs_give_each_member_its_own_initial_state():
    source = member_inputs_source()
    assert source.member_input_config(1, T0 + 12 * HOUR) == {"dataset": "/m2/2024010212.zarr"}
    frames = list(source.frames(T0 + 12 * HOUR, 12 * HOUR, ["2t", "10u"], CPU))
    assert [f.lead_time for f in frames] == [6 * HOUR, 12 * HOUR] and frames[0].variables == ["2t", "10u"]
    for k, frame in enumerate(frames, start=1):
        assert frame.data.shape == (2, 2, N) and frame.data.dtype == torch.float32
        for m in range(2):
            torch.testing.assert_close(frame.data[m, 0], torch.full((N,), 12.0 + m + k))
            torch.testing.assert_close(frame.data[m, 1], torch.full((N,), 3.0 * (m + 1) + k))
    lead0 = source.initial_frame(T0, ["2t", "10u"], CPU)
    torch.testing.assert_close(lead0.data[:, 0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(lead0.data[:, 1, 0], torch.tensor([3.0, 6.0]))


def test_member_inputs_are_opened_once(monkeypatch):
    import anemoi.inference.inputs
    import anemoi.inference.variables

    opened = []

    def create_input(runner, config, metadata, variables, purpose):
        opened.append(config)
        return SimpleNamespace(config=config)

    monkeypatch.setattr(anemoi.inference.inputs, "create_input", create_input)
    monkeypatch.setattr(
        anemoi.inference.variables, "Variables", lambda metadata: SimpleNamespace(retrieved_prognostic_variables=list)
    )
    source = member_inputs_source()
    del source._member_input  # the real method, not the fake
    first = source._member_input(0, T0)
    assert source._member_input(0, T0) is first and source._member_input(1, T0) is not first
    assert source._member_input(0, T0 + 24 * HOUR).config == {"dataset": "/m1/2024010300.zarr"}
    assert len(opened) == 3


def test_boundary_forcings_dates_are_checked_only_when_the_runner_reads_them(monkeypatch):
    """A checkpoint without an output_mask has no boundary-forcing providers: its boundary input is never read, so
    a +24h rollout from an input that only covers the initial window must pass the date check."""
    import anemoi.inference.inputs.dataset
    import anemoi.inference.inputs.empty

    class FakeDatasetInput:
        def __init__(self, dates):
            self.ds_dates = np.array(dates, dtype="datetime64[us]")
            self.ds = SimpleNamespace(missing=set())

    class FakeEmptyInput:
        pass

    monkeypatch.setattr(anemoi.inference.inputs.dataset, "DatasetInput", FakeDatasetInput)
    monkeypatch.setattr(anemoi.inference.inputs.empty, "EmptyInput", FakeEmptyInput)
    source = member_inputs_source()
    source.member_inputs = None
    source.input_offsets = [-6 * HOUR, 0 * HOUR]
    short = FakeDatasetInput([T0 - 6 * HOUR, T0, T0 + 6 * HOUR])  # the initial window, not a +24h rollout
    source.runner.prognostics_inputs = {"ds": short}
    source.runner.dynamic_forcings_inputs = {"ds": FakeEmptyInput()}
    source.runner.boundary_forcings_inputs = {"ds": short}
    source.runner.tensor_handlers = {"ds": SimpleNamespace(boundary_forcings_providers=[])}
    source.check_init_times([T0], 24 * HOUR)  # passes: no output_mask, boundaries never read
    source.runner.tensor_handlers = {"ds": SimpleNamespace(boundary_forcings_providers=["boundary"])}
    with pytest.raises(ValueError, match="not in the boundary forcings dataset"):
        source.check_init_times([T0], 24 * HOUR)
    assert source._reads_boundary_forcings("ds")
    unknown = member_inputs_source()
    del unknown.runner.tensor_handlers
    assert unknown._reads_boundary_forcings("ds")  # no tensor handler to ask: unknown counts as yes


def test_member_inputs_of_other_types_are_opened_but_not_date_checked(monkeypatch, caplog):
    import logging

    import anemoi.inference.inputs.dataset

    class FakeDatasetInput:
        def __init__(self, dates):
            self.ds_dates = np.array(dates, dtype="datetime64[us]")
            self.ds = SimpleNamespace(missing=set())

    monkeypatch.setattr(anemoi.inference.inputs.dataset, "DatasetInput", FakeDatasetInput)
    source = member_inputs_source()
    source.input_offsets = [-6 * HOUR, 0 * HOUR]
    other = SimpleNamespace()  # e.g. a GRIB input: opened, its dates unknown to the check
    source._member_input = lambda member, init_time: other if member == 0 else FakeDatasetInput([T0 - 6 * HOUR, T0])
    with caplog.at_level(logging.WARNING):
        source._check_member_inputs([T0], 6 * HOUR)
    assert "SimpleNamespace" in caplog.text and "not anemoi datasets" in caplog.text
    source._member_input = lambda member, init_time: FakeDatasetInput([T0])  # the t-6h input date is missing
    with pytest.raises(ValueError, match="member 0: dates"):
        source._check_member_inputs([T0], 6 * HOUR)


def test_member_inputs_give_each_member_its_own_boundary_forcings(monkeypatch):
    """A checkpoint with boundary forcings (an output_mask) reads them at every step from the tensor handler's
    providers: each member must advance with the providers built from its own input, which must cover the rollout."""
    source = member_inputs_source()
    seen = []

    class FakeHandler:
        boundary_forcings_input = "runner input"
        boundary_forcings_providers = ["runner boundaries"]

        def create_boundary_forcings_providers(self):
            return [("boundaries", self.boundary_forcings_input.member, self.boundary_forcings_input.init_time)]

    handler = FakeHandler()
    source.runner.tensor_handlers = {"ds": handler}

    def run(input_states, lead_time, return_numpy):
        state = input_states["ds"]
        for k in range(1, int(lead_time / (6 * HOUR)) + 1):
            seen.append(list(handler.boundary_forcings_providers))  # what this step's boundaries are read from
            fields = {v: torch.as_tensor(x[-1]) + k for v, x in state["fields"].items()}
            yield {"ds": {"step": 6 * k * HOUR, "date": state["date"] + 6 * k * HOUR, "fields": fields}}

    source.runner.run = run
    frames = list(source.frames(T0, 12 * HOUR, ["2t"], CPU))
    assert len(frames) == 2
    assert seen == [[("boundaries", m, T0)] for _ in range(2) for m in range(2)]  # step by step, member by member
    assert handler.boundary_forcings_providers == ["runner boundaries"]
    assert handler.boundary_forcings_input == "runner input"

    import anemoi.inference.inputs.dataset

    class FakeDatasetInput:
        def __init__(self, dates):
            self.ds_dates = np.array(dates, dtype="datetime64[us]")
            self.ds = SimpleNamespace(missing=set())

    monkeypatch.setattr(anemoi.inference.inputs.dataset, "DatasetInput", FakeDatasetInput)
    source.input_offsets = [-6 * HOUR, 0 * HOUR]
    source._member_input = lambda member, init_time: FakeDatasetInput([T0 - 6 * HOUR, T0, T0 + 6 * HOUR])
    source._check_member_inputs([T0], 12 * HOUR)  # boundaries at +6h, read before the second call: covered
    with pytest.raises(ValueError, match="member 0: dates"):
        source._check_member_inputs([T0], 18 * HOUR)  # +12h boundaries missing
    handler.boundary_forcings_providers = []  # no output_mask: only the input window is needed
    source._check_member_inputs([T0], 18 * HOUR)
