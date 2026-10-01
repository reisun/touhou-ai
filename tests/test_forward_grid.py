import numpy as np
import torch
from touhou_ai.forward_grid import ForwardEncoder, ForwardEnv, build
from touhou_ai.spatial_input_candidates import CandidateEncoder


def test_control_and_detail_geometry():
    environment = ForwardEnv()
    environment.reset(seed=7)
    raw = environment.raw_observation()
    raw['player']['position'] = [0, 350]
    raw['bullets'] = [dict(position=[0, 220], velocity_raw=[0, 1], hitbox_raw=[4, 4], flags_raw=2)]
    expected = CandidateEncoder('action_grid').encode(raw)
    control = ForwardEncoder('control').encode(raw)
    for key in expected:
        np.testing.assert_array_equal(control[key], expected[key])
    detail = ForwardEncoder('detail').encode(raw)
    assert detail['local_grid'][1].sum() > 0
    assert control['local_grid'][1].sum() == 0
    assert np.nonzero(detail['local_grid'][0])[0].mean() == 71.5
    for key in control:
        if key != 'local_grid':
            np.testing.assert_array_equal(control[key], detail[key])
    raw['bullets'] *= 32
    dense_control = ForwardEncoder('control').encode(raw)
    dense_detail = ForwardEncoder('detail').encode(raw)
    np.testing.assert_array_equal(dense_control['global_grid'], dense_detail['global_grid'])


def test_wide_forward_boundary_and_other_channels():
    environment = ForwardEnv()
    environment.reset(seed=7)
    raw = environment.raw_observation()
    raw['player']['position'] = [0, 350]
    raw['bullets'] = [dict(position=[0, 200], velocity_raw=[0, 2], hitbox_raw=[4, 4], flags_raw=2)]
    baseline = ForwardEncoder('control').encode(raw)
    wide = ForwardEncoder('wide').encode(raw)
    np.testing.assert_array_equal(wide['global_grid'], baseline['global_grid'][3:])
    np.testing.assert_array_equal(wide['local_grid'], baseline['local_grid'])
    assert wide['wide_grid'][0].sum() == 1
    assert np.nonzero(wide['wide_grid'][1])[0].mean() > np.nonzero(wide['wide_grid'][0])[0].mean()
    raw['bullets'][0]['position'] = [0, 270]
    assert ForwardEncoder('wide').encode(raw)['wide_grid'].sum() == 0


def test_initial_parameters_rng_and_physics():
    torch.set_num_threads(1)
    baseline, _, _ = build('control', 7)
    rng = torch.get_rng_state()
    candidate, _, _ = build('both', 7)
    assert torch.equal(rng, torch.get_rng_state())
    original = baseline.policy.state_dict()
    for key, value in candidate.policy.state_dict().items():
        if key in original and value.shape == original[key].shape:
            assert torch.equal(value, original[key])
    environments = [ForwardEnv(variant) for variant in ('control', 'detail', 'wide', 'both')]
    for environment in environments:
        environment.reset(seed=55)
    for frame in range(100):
        outcomes = [environment.step([0, 0, frame % 2, 0]) for environment in environments]
        assert all(outcome[1:4] == outcomes[0][1:4] for outcome in outcomes)
        for environment in environments[1:]:
            np.testing.assert_array_equal(environment.xy, environments[0].xy)
        if outcomes[0][2]:
            break
