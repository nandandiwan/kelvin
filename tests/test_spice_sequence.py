import json
from pathlib import Path

import pytest

from gds.spice_sequence import run_sequence, validate_schedule, write_sequence_deck
from gds.spice_power import find_ngspice


@pytest.fixture
def schedule():
    return json.loads((Path(__file__).resolve().parents[1] / "data/sequence_2x2.json").read_text())


def test_masked_state_carries_between_operations(schedule):
    _, records = validate_schedule(schedule)
    assert records[2]["before"] == [[1, 0], [1, 0]]
    assert records[4]["before"] == [[1, 0], [0, 0]]
    assert records[-1]["after"] == [[1, 0], [0, 1]]


@pytest.mark.parametrize("change", ["overlap", "bad_row", "bad_mask", "bad_bits", "bad_cap"])
def test_invalid_sequence_rejected(schedule, change):
    if change == "overlap":
        schedule["operations"][1]["start_ns"] = .2
    elif change == "bad_row":
        schedule["operations"][0]["row"] = 2
    elif change == "bad_mask":
        schedule["operations"][1]["mask"] = [1]
    elif change == "bad_bits":
        schedule["initial_bits"][0][0] = 2
    else:
        schedule["bitline_cap_f"] = -1
    with pytest.raises(ValueError):
        validate_schedule(schedule)


def test_one_continuous_transient_and_one_load_per_column(schedule, tmp_path):
    write_sequence_deck(tmp_path, schedule)
    deck = (tmp_path / "sequence.sp").read_text()
    assert deck.count(".tran ") == 1
    assert deck.count(".nodeset") == 4
    assert deck.count("CBL") == 2
    assert deck.count("CBR") == 2
    assert ".temp 26.85" in deck
    with pytest.raises(FileExistsError):
        write_sequence_deck(tmp_path, schedule)


@pytest.mark.parametrize("temperature,step", [(26.85, 1), (31.85, 1), (26.85, .5)])
def test_real_ngspice_stateful_shared_array(schedule, tmp_path, temperature, step):
    try:
        find_ngspice()
    except FileNotFoundError:
        pytest.skip("ngspice unavailable")
    report = run_sequence(tmp_path, schedule, temperature_c=temperature, max_step_ps=step)
    assert report["completed"]
    assert report["cell_count"] == 4
    assert report["operation_count"] == 7
    assert report["channel_energy_j"] > 0
    assert len(report["per_device_energy_j"]) == 32


def test_electrical_timestep_energy_convergence(schedule, tmp_path):
    try:
        find_ngspice()
    except FileNotFoundError:
        pytest.skip("ngspice unavailable")
    coarse = run_sequence(tmp_path / "coarse", schedule, max_step_ps=1)
    fine = run_sequence(tmp_path / "fine", schedule, max_step_ps=.5)
    assert abs(coarse["channel_energy_j"] / fine["channel_energy_j"] - 1) < 1e-3
    # Check individual heat sources too; use total energy for tiny/zero sources.
    for identity, energy in fine["per_device_energy_j"].items():
        assert abs(coarse["per_device_energy_j"][identity] - energy) < 1e-4 * fine["channel_energy_j"]
