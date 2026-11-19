"""Temporal downscalers between analysis bounds, and their linear-interpolation baseline.

The downscaler runs on anemoi-inference's real `TemporalDownscalerMultiOutRunner` over the multi-dataset fixture's
checkpoint, patched into a downscaler with bounds `0` and `3` data steps (a window of 18 h at the fixture's 6 h) and
outputs at `1` and `2`; only its model is a fake: it interpolates the two input bounds linearly and adds `noise` times
a draw of the global RNG, so a test can tell the members, the seeding and the placement of the outputs apart.
"""

import datetime
import logging

import _multi_dataset as fixture
import numpy as np
import pytest
import torch

from anemoi.evaluation.evaluate import Evaluation
from anemoi.evaluation.evaluate import build
from anemoi.evaluation.frame import Grid
from anemoi.evaluation.sources.anemoi_dataset import DatasetTargets
from anemoi.evaluation.sources.anemoi_inference import InferenceForecastSource
from anemoi.evaluation.sources.anemoi_inference import member_seed
from anemoi.evaluation.sources.fake import ArrayTargets
from anemoi.evaluation.sources.linear_interpolation import LinearInterpolationForecastSource
from anemoi.evaluation.sources.temporal_downscaler import downscaler_window

HOUR = datetime.timedelta(hours=1)
WINDOW = 18 * HOUR
T0 = datetime.datetime(2020, 1, 2)


def downscaler_patch(targets=(1, 2), inputs=(0, 3)):
    """A `save_checkpoint` patch turning the fixture forecaster into a temporal downscaler, stored the way the dummy
    anemoi-core downscaler checkpoint stores it (no output offsets, `timestep` the window)."""

    def patch(metadata):
        metadata["metadata_inference"]["task"] = "temporal-downscaler"
        metadata["config"]["task"] = {
            "_target_": "anemoi.training.tasks.TemporalDownscaler",
            "input_timestep": f"{6 * (inputs[1] - inputs[0])}h",
            "output_timestep": "6h",
        }
        # the runner builds its default inputs from the test dataloader; the tests replace them
        metadata["config"]["dataloader"]["test"] = metadata["config"]["dataloader"]["validation"]
        for name in metadata["metadata_inference"]["dataset_names"]:
            metadata["metadata_inference"][name]["timesteps"] = {
                "relative_date_indices_training": list(range(inputs[0], inputs[1] + 1)),
                "input_relative_date_indices": list(inputs),
                "output_relative_date_indices": list(targets),
                "timestep": f"{6 * (inputs[1] - inputs[0])}h",
            }

    return patch


def interpolating_model(runner, noise):
    """A `predict_step` linearly interpolating each output variable present in the input between the two bounds
    (the diagnostics get the output index), plus `noise` times a standard normal draw of the output's shape; it
    counts its calls in `calls`."""
    n = len(runner.checkpoint.target_explicit_times)
    inputs = runner.checkpoint.input_explicit_times
    weights = [(t - inputs[0]) / (inputs[1] - inputs[0]) for t in runner.checkpoint.target_explicit_times]

    def predict_step(model, input_tensors, **kwargs):
        predict_step.calls += 1
        outputs = {}
        for name, x in input_tensors.items():  # (1, 2, values, input variables)
            metadata = runner.checkpoint.multi_dataset_metadata[name]
            index = metadata.variable_to_input_tensor_index
            variables = metadata.output_tensor_index_to_variable
            y = torch.zeros(1, n, 1, x.shape[2], len(variables), dtype=torch.float32)
            for s, a in enumerate(weights):
                for i, variable in variables.items():
                    if variable in index:
                        y[0, s, 0, :, i] = (1 - a) * x[0, 0, :, index[variable]] + a * x[0, 1, :, index[variable]]
                    else:
                        y[0, s, 0, :, i] = s + 1
            outputs[name] = y + noise * torch.randn(y.shape)
        return outputs

    predict_step.calls = 0
    return predict_step


def downscaler_source(
    monkeypatch,
    tmp_path,
    names=("era5",),
    targets=(1, 2),
    noise=0.0,
    write_initial_state=True,
    patch=None,
    runner_class=None,
    **options,
):
    """An `InferenceForecastSource` over a fixture downscaler checkpoint, driven by the real downscaler runner (or
    by `runner_class`, built like the fixture's forecaster runner); `patch` edits the metadata further."""

    def patches(metadata):
        downscaler_patch(targets)(metadata)
        if patch is not None:
            patch(metadata)

    checkpoint = fixture.save_checkpoint(tmp_path / "td.ckpt", names, patch=patches)
    patch_downscaler_runner(monkeypatch, checkpoint, names, noise, write_initial_state, runner_class)
    return InferenceForecastSource(checkpoint=str(checkpoint), device="cpu", **options)


def patch_downscaler_runner(
    monkeypatch, checkpoint, names=("era5",), noise=0.0, write_initial_state=True, runner_class=None
):
    """Make `InferenceForecastSource` build the downscaler runner of `downscaler_source` over `checkpoint`."""
    import anemoi.inference.runners as runners
    from anemoi.inference.inputs.empty import EmptyInput
    from anemoi.inference.runners.temporal_downscaler import TemporalDownscalerMultiOutRunner

    input_class = fixture._fixture_input_class()

    def create_runner(config):
        if runner_class is not None:
            return fixture.make_runner(checkpoint, names)
        runner = TemporalDownscalerMultiOutRunner(config)
        for name in names:
            metadata = runner.checkpoint.multi_dataset_metadata[name]
            dataset = fixture.FixtureDataset(name)
            constants = metadata.select_variables(include=["constant"], exclude=["computed"])
            others = [v for v in metadata.variable_to_input_tensor_index if v not in constants]
            runner.prognostics_inputs[name] = input_class(runner, dataset, metadata, others)
            runner.constant_forcings_inputs[name] = input_class(runner, dataset, metadata, constants)
            runner.dynamic_forcings_inputs[name] = EmptyInput(runner, metadata)
            runner.boundary_forcings_inputs[name] = EmptyInput(runner, metadata)
        runner.predict_step = interpolating_model(runner, noise)
        runner.write_initial_state = write_initial_state
        return runner

    monkeypatch.setattr(runners, "create_runner", create_runner)


def test_a_downscaler_is_scored_one_window_per_init_time(monkeypatch, tmp_path):
    source = downscaler_source(monkeypatch, tmp_path)
    assert source.temporal_downscaling and not source.supports_lead_zero
    assert (source.window, source.window_leads, source.frames_per_pass) == (WINDOW, [6 * HOUR, 12 * HOUR], 2)
    assert source.lead_times(WINDOW) == [6 * HOUR, 12 * HOUR]
    with pytest.raises(ValueError, match="the lead time must be its window 18h, got 12h"):
        source.lead_times(12 * HOUR)
    with pytest.raises(ValueError, match="no lead-0 frame"):
        source.initial_frame(T0, ["2t"], torch.device("cpu"))
    info = source.describe()
    assert (info["task"], info["window"]) == ("temporal-downscaler", "18h")
    targets = fixture.make_targets("era5")
    with pytest.raises(ValueError, match="cannot produce lead-0 frames"):
        Evaluation(source, targets, [T0], WINDOW, ["rmse"], include_lead_zero=True)
    evaluation = Evaluation(source, targets, [T0], WINDOW, ["rmse"])
    assert evaluation.lead_times == [6 * HOUR, 12 * HOUR] and evaluation.model_calls_per_init() == (1, WINDOW)


def test_the_runner_must_fit_the_task(monkeypatch, tmp_path):
    """A downscaler driven by a forecaster runner (and the reverse) is refused, rather than scoring nothing."""
    with pytest.raises(
        ValueError, match=r"task is 'temporal-downscaler' but anemoi-inference built a SimpleRunner: set"
    ):
        downscaler_source(monkeypatch, tmp_path, runner_class="forecaster")

    def only_in_the_task_class(metadata):  # what anemoi-inference's `auto` runner does not read
        metadata["metadata_inference"]["task"] = None

    with pytest.raises(
        ValueError,
        match=r"`runner: auto` reads only `metadata_inference.task`, which is None.*"
        r"`runner: temporal_downscaler`",
    ):
        downscaler_source(monkeypatch, tmp_path, patch=only_in_the_task_class, runner_class="forecaster")

    def a_forecaster(metadata):
        metadata["metadata_inference"]["task"] = "forecaster"

    with pytest.raises(
        ValueError, match=r"task is 'forecaster' but .* TemporalDownscalerMultiOutRunner: set `runner: default`"
    ):
        downscaler_source(monkeypatch, tmp_path, patch=a_forecaster)


def test_steps_other_than_the_window_leads_raise(monkeypatch, tmp_path):
    source = downscaler_source(monkeypatch, tmp_path)
    source.window_leads = [6 * HOUR]  # the runner yields 6h and 12h
    with pytest.raises(RuntimeError, match="the runner yielded 12:00:00 at 2020-01-02 00:00:00, not a lead"):
        list(source.frames(T0, WINDOW, ["2t"], torch.device("cpu")))
    source.window_leads = [6 * HOUR, 12 * HOUR, WINDOW]  # one the runner does not yield
    with pytest.raises(RuntimeError, match=r"yielded the steps \[.*\] at 2020-01-02 00:00:00, the window's leads"):
        list(source.frames(T0, WINDOW, ["2t"], torch.device("cpu")))


@pytest.mark.parametrize("write_initial_state", [True, False])
def test_the_frames_are_the_outputs_inside_the_window(monkeypatch, tmp_path, write_initial_state):
    """The bounds are skipped, the outputs land on their valid times as tensors, and a model that interpolates
    linearly scores what the linear-interpolation baseline scores."""
    source = downscaler_source(monkeypatch, tmp_path, write_initial_state=write_initial_state)
    variables = ["2t", "msl", "tp"]
    frames = list(source.frames(T0, WINDOW, variables, torch.device("cpu")))
    assert [(f.lead_time, f.valid_time) for f in frames] == [(6 * HOUR, T0 + 6 * HOUR), (12 * HOUR, T0 + 12 * HOUR)]
    assert all(isinstance(f.data, torch.Tensor) and f.data.shape == (1, 3, 288) for f in frames)
    np.testing.assert_array_equal(frames[1].data[0, 2], 2.0)  # the diagnostic, by output index
    baseline = LinearInterpolationForecastSource(fixture.make_targets("era5"), "6h")
    expected = list(baseline.frames(T0, WINDOW, variables[:2], torch.device("cpu")))
    for frame, reference in zip(frames, expected):
        torch.testing.assert_close(frame.data[:, :2], reference.data)


@pytest.mark.parametrize("write_initial_state", [True, False])
def test_lockstep_members_are_seeded_per_member_before_their_model_call(monkeypatch, tmp_path, write_initial_state):
    seed, variables, cpu = 7, ["2t", "q_850"], torch.device("cpu")
    options = {"write_initial_state": write_initial_state, "seed": seed}
    linear = [f.data for f in downscaler_source(monkeypatch, tmp_path).frames(T0, WINDOW, variables, cpu)]
    source = downscaler_source(monkeypatch, tmp_path, noise=1.0, members=3, **options)
    frames = list(source.frames(T0, WINDOW, variables, cpu))
    assert [f.data.shape for f in frames] == [(3, 2, 288)] * 2 and source.runner.predict_step.calls == 3
    output_variables = source.runner.checkpoint.multi_dataset_metadata["era5"].output_tensor_index_to_variable
    columns = [next(i for i, v in output_variables.items() if v == name) for name in variables]
    for member in range(3):
        torch.manual_seed(member_seed(seed, T0, member, 0))
        draw = torch.randn(1, 2, 1, 288, len(output_variables))
        for s, frame in enumerate(frames):
            expected = linear[s][0] + draw[0, s, 0][:, columns].T  # what the model adds, in float32
            torch.testing.assert_close(frame.data[member], expected, rtol=0, atol=0)
    # a member does not depend on the ensemble size
    single = list(downscaler_source(monkeypatch, tmp_path, noise=1.0, **options).frames(T0, WINDOW, variables, cpu))
    for frame, alone in zip(frames, single):
        torch.testing.assert_close(frame.data[:1], alone.data, rtol=0, atol=0)


def test_two_datasets_share_one_window_call_and_shard(monkeypatch, tmp_path):
    source = downscaler_source(monkeypatch, tmp_path, names=fixture.DATASET_NAMES)
    config = {
        "forecast": {"anemoi_inference": {"checkpoint": "unused"}},
        "lead_time": "18h",
        "init_times": {"dates": [T0.isoformat(), (T0 + 6 * HOUR).isoformat()]},
        "variables": {"datasets": {"era5": ["2t"], "realch1": ["2t", "msl"]}},
        "weights": {"uniform": {}},
    }
    targets = {name: fixture.make_targets(name) for name in fixture.DATASET_NAMES}
    evaluation = build(config, forecast=source, targets=targets)
    whole = evaluation.run()
    assert source.runner.predict_step.calls == 2  # one call per init time for both datasets
    assert all(state.lead_times == [6 * HOUR, 12 * HOUR] for state in whole.values())
    first, second = (evaluation.run(evaluation.shard(i, 2)) for i in range(2))
    for name, state in whole.items():
        merged = first[name].merge(second[name])
        for key in state.sums:
            torch.testing.assert_close(merged.sums[key], state.sums[key], rtol=0, atol=0)


def test_a_right_bound_output_is_scored_and_a_left_one_refused(monkeypatch, tmp_path):
    source = downscaler_source(monkeypatch, tmp_path, targets=(1, 2, 3))
    assert source.lead_times(WINDOW) == [6 * HOUR, 12 * HOUR, WINDOW]
    frames = list(source.frames(T0, WINDOW, ["2t"], torch.device("cpu")))
    assert [f.lead_time for f in frames] == [6 * HOUR, 12 * HOUR, WINDOW]
    right = fixture.make_targets("era5").frame(T0 + WINDOW, ["2t"], torch.device("cpu"))
    torch.testing.assert_close(frames[-1].data, right)
    with pytest.raises(
        ValueError, match=r"target times \[0, 1, 2\].*not supported yet.*labels the outputs 6h, 12h, 18h"
    ):
        downscaler_source(monkeypatch, tmp_path, targets=(0, 1, 2))
    checkpoint = type("Checkpoint", (), {"input_explicit_times": [0, 6], "target_explicit_times": [1, 2, 3, 4, 5]})
    checkpoint.data_frequency = "1h"
    assert downscaler_window(checkpoint) == (6 * HOUR, [k * HOUR for k in range(1, 6)])


def test_init_times_need_both_bounds(monkeypatch, tmp_path):
    source = downscaler_source(monkeypatch, tmp_path)
    last = datetime.datetime(2020, 1, 5)
    dataset = source.runner.prognostics_inputs["era5"].ds
    dataset.missing = {dataset.index(T0 + 6 * HOUR)}  # inside the window: not an input
    source.check_init_times([T0, last - WINDOW], WINDOW)
    with pytest.raises(ValueError, match=r"init time 2020-01-04 12:00:00: dates \[datetime.datetime\(2020, 1, 5, 6"):
        source.check_init_times([last - 12 * HOUR], WINDOW)
    dataset.missing = {dataset.index(T0 + WINDOW)}
    with pytest.raises(ValueError, match="are not in the prognostics dataset"):
        source.check_init_times([T0], WINDOW)


def test_interpolated_targets_are_warned_about(monkeypatch, tmp_path, caplog):
    source = downscaler_source(monkeypatch, tmp_path, names=fixture.DATASET_NAMES)
    view = source.datasets["era5"]  # the flag reaches a multi-dataset view through its delegation
    plain = fixture.make_targets("era5")
    interpolated = DatasetTargets.from_dataset(
        fixture.FixtureDataset("era5"),
        kwargs={"dataset": {"cutout": [{"dataset": "lam"}, {"dataset": "global", "interpolate_frequency": "1h"}]}},
    )
    assert plain.interpolate_frequency == [] and interpolated.interpolate_frequency == ["1h"]
    with caplog.at_level(logging.WARNING):
        Evaluation(view, plain, [T0], WINDOW, ["rmse"])
        assert "interpolate_frequency" not in caplog.text
        Evaluation(view, interpolated, [T0], WINDOW, ["rmse"])
        assert "opened with interpolate_frequency ['1h']" in caplog.text
        caplog.clear()
        baseline = LinearInterpolationForecastSource(interpolated, "6h")
        Evaluation(baseline, interpolated, [T0], WINDOW, ["rmse"])
        assert "opened with interpolate_frequency ['1h']" in caplog.text


def test_a_downscaler_run_matches_its_linear_baseline(monkeypatch, tmp_path):
    """End to end on the fixture: the interpolating model and the baseline score the same, leads 6h and 12h only."""
    source = downscaler_source(monkeypatch, tmp_path)
    targets = fixture.make_targets("era5")
    init_times, options = [T0, T0 + 6 * HOUR], {"variables": ["2t", "10u"], "weights": {"uniform": {}}}
    model = Evaluation(source, targets, init_times, WINDOW, ["rmse", "bias"], **options).run()
    baseline = LinearInterpolationForecastSource(targets, "6h")
    reference = Evaluation(baseline, targets, init_times, WINDOW, ["rmse", "bias"], **options).run()
    assert model.lead_times == reference.lead_times == [6 * HOUR, 12 * HOUR]
    for key in model.sums:
        # the model interpolates in float32, the baseline in float64
        np.testing.assert_allclose(model.sums[key], reference.sums[key], rtol=1e-4, atol=1e-3)


# linear interpolation baseline


class KnownTargets(ArrayTargets):
    """Array targets that know which valid times they have."""

    def available(self, valid_time):
        return valid_time in self.fields


def interpolation_targets(n=5, variables=("a", "b"), hours=(0, 6, 12, 18, 24), seed=0):
    rng = np.random.default_rng(seed)
    grid = Grid(np.linspace(-80, 80, n), np.linspace(0, 350, n))
    fields = {T0 + h * HOUR: rng.normal(size=(len(variables), n)).astype(np.float32) for h in hours}
    return KnownTargets(grid, list(variables), fields), fields


def test_linear_interpolation_against_a_hand_computation():
    targets, fields = interpolation_targets()
    source = LinearInterpolationForecastSource(targets, "6h", members=2)
    assert source.temporal_downscaling and not source.supports_lead_zero
    assert source.lead_times(24 * HOUR) == [6 * HOUR, 12 * HOUR, 18 * HOUR]
    frames = list(source.frames(T0, 24 * HOUR, ["b", "a"], torch.device("cpu")))
    assert [(f.lead_time, f.valid_time) for f in frames] == [(k * HOUR, T0 + k * HOUR) for k in (6, 12, 18)]
    left, right = fields[T0][[1, 0]].astype(np.float64), fields[T0 + 24 * HOUR][[1, 0]].astype(np.float64)
    for frame, a in zip(frames, (0.25, 0.5, 0.75)):
        expected = ((1 - a) * left + a * right).astype(np.float32)
        assert frame.data.dtype == torch.float32 and frame.data.shape == (2, 2, 5)
        np.testing.assert_array_equal(frame.data[0].numpy(), expected)
        np.testing.assert_array_equal(frame.data[1].numpy(), expected)
    # one window of 12 h from 06: halfway is 12
    (middle,) = source.frames(T0 + 6 * HOUR, 12 * HOUR, ["a"], torch.device("cpu"))
    halfway = (0.5 * fields[T0 + 6 * HOUR][0].astype(np.float64) + 0.5 * fields[T0 + 18 * HOUR][0]).astype(np.float32)
    np.testing.assert_array_equal(middle.data[0, 0].numpy(), halfway)
    with pytest.raises(ValueError, match="no time strictly inside it"):
        source.lead_times(6 * HOUR)
    with pytest.raises(ValueError, match="not a positive multiple"):
        source.lead_times(9 * HOUR)


def test_linear_interpolation_skips_windows_without_both_bounds(caplog):
    targets, _ = interpolation_targets(hours=(0, 6, 12, 18))
    source = LinearInterpolationForecastSource(targets, "6h")
    with caplog.at_level(logging.WARNING):
        assert list(source.frames(T0, 24 * HOUR, ["a"], torch.device("cpu"))) == []
    assert "no linear interpolation from 2020-01-02" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        source.check_init_times([T0, T0 + 6 * HOUR], 12 * HOUR)
        assert not caplog.records
        source.check_init_times([T0, T0 + 6 * HOUR, T0 + 12 * HOUR], 12 * HOUR)
    assert "1 init times have no target at a bound of their window" in caplog.text


def test_linear_interpolation_from_the_config():
    targets, fields = interpolation_targets()
    config = {
        "init_times": {"dates": [T0.isoformat()]},
        "lead_time": "24h",
        "forecast": {"linear_interpolation": {"timestep": "6h", "members": 2}},
        "metrics": ["rmse", "crps"],
        "weights": {"uniform": {}},
    }
    evaluation = build(config, targets=targets)
    assert isinstance(evaluation.forecast, LinearInterpolationForecastSource)
    assert evaluation.lead_times == [6 * HOUR, 12 * HOUR, 18 * HOUR]
    assert evaluation.to_config()["forecast"] == {"linear_interpolation": {"timestep": "6h", "members": 2}}
    state = evaluation.run()
    result = state.to_xarray(evaluation.metrics)
    left, right = fields[T0].astype(np.float64), fields[T0 + 24 * HOUR].astype(np.float64)
    expected = ((0.5 * left + 0.5 * right).astype(np.float32) - fields[T0 + 12 * HOUR]).astype(np.float64)
    rmse = np.sqrt((expected**2).mean(1))
    got = result["rmse"].sel(lead_time=np.timedelta64(12, "h"), bin="all", region="global")
    np.testing.assert_allclose(got.sel(variable=["a", "b"]).values, rmse, rtol=1e-6)
