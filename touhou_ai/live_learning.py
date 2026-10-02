"""Bounded on-policy learning from the real game; staged, explicitly partial contract."""
from touhou_ai.vibration_reward import VibrationReward, SOURCE as JITTER_SOURCE, SPEC as JITTER_SPEC
from contextlib import nullcontext
from touhou_ai.ui_stats import UiStats
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import configure

from touhou_ai.game_rewards import EventRewards
from touhou_ai.live_runtime import LiveRuntime, input_mask
from touhou_ai.live_reset import start_episode, continue_episode, wait_game_over, GameOverHold, advance_dialogue
from touhou_ai.policy_check import NumericalFeatures
from touhou_ai.telemetry import packet
from touhou_ai.live_transition import bridge_transition, CollectionInterrupted
from touhou_ai.live_rewards import LiveRewards, VERSION as REWARD_VERSION, WEIGHTS as REWARD_WEIGHTS, ENABLED as REWARD_ENABLED, observed_events
from touhou_ai.bullet_scope import scoped, SPEC as BULLET_SCOPE
from touhou_ai.timed_bomb import TimedBombPolicy, BombClock, SPEC as BOMB_SCHEDULE
from touhou_ai.learning_discount import SPEC as DISCOUNT_CONTRACT, verify_settings


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "th10-live-observed-v1"
CAPACITIES = {"bullets": BULLET_SCOPE['individual_bullets'], "enemies": 24, "items": 40, "lasers": 64}


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01)


def publish_telemetry(data):
    from touhou_ai.telemetry_memory import publish
    return publish(data, ROOT / 'artifacts')


class ObservedContract(gym.Env):
    """Only measured features. Availability masks distinguish absent managers.

    Deliberately separate from the full Sharu encoder: no invented HP, IDs,
    acceleration, elapsed-hit/bomb state or calibrated collision radii.
    """
    def __init__(self, include_grid=True):
        self.observation_space = gym.spaces.Dict({
            "player": gym.spaces.Box(-1, 1, (15,), dtype=np.float32),
            "far_grid": gym.spaces.Box(-1, 1, (3, 112, 96), dtype=np.float32),
            **{k: gym.spaces.Box(-1, 1, (n, 5), dtype=np.float32) for k, n in CAPACITIES.items()}})
        # Bomb is a one-choice head: disabling it must not corrupt PPO log probabilities.
        self.action_space = gym.spaces.MultiDiscrete([9, 2, 2, 1])
        if not include_grid:
            del self.observation_space.spaces['far_grid']

    def reset(self, **kwargs):
        raise RuntimeError("real collector owns reset")

    def step(self, action):
        raise RuntimeError("real collector owns stepping")

    def encode(self, raw):
        raw = scoped(raw)
        if raw["player"] is None:
            raise ValueError("player missing")
        if raw.get("lasers"):
            raise ValueError("live laser geometry has not passed acceptance")
        out = {k: np.zeros(s.shape, dtype=np.float32) for k, s in self.observation_space.spaces.items()}
        x, y = raw["player"]["position"]
        player = raw["player"]
        out["player"][:] = [x/192, y/448, player["velocity_raw"][0]/1000,
            player["velocity_raw"][1]/1000, raw["lives_raw"]/8, raw["power_raw"]/100,
            player["status"]/4, player["invincibility_raw"]/300, player["focus_raw"],
            *[raw.get(k) is not None for k in CAPACITIES], raw["stage"]/6, 1]
        for kind, capacity in CAPACITIES.items():
            entities = raw.get(kind)
            if entities is None:
                continue
            # No temporal identity is claimed; permutation-invariant pooling uses this frame only.
            ordered = sorted(entities, key=lambda e: ((e["position"][0]-x)**2+(e["position"][1]-y)**2,
                                                      *e["position"], *e["velocity_raw"]))
            for index, entity in enumerate(ordered):
                ex, ey = entity["position"]
                vx, vy = entity["velocity_raw"]
                if index < capacity:
                    out[kind][index] = [1, (ex-x)/384, (ey-y)/448, vx/10, vy/10]
                elif kind == "bullets" and 'far_grid' in out and 0 <= ex+192 < 384 and 0 <= ey < 448:
                    out["far_grid"][:, int(ey//4), int((ex+192)//4)] += [1, vx/10, vy/10]
        if 'far_grid' in out:
            counts = out["far_grid"][0].copy()
            out["far_grid"][1:] /= np.maximum(counts, 1)
            out["far_grid"][0] = np.log1p(counts)/np.log(17)
        for array in out.values():
            if not np.isfinite(array).all():
                raise ValueError("nonfinite live observation")
            np.clip(array, -1, 1, out=array)
        return out


class ExtendedObservedContract(ObservedContract):
    def __init__(self, include_grid=True):
        from touhou_ai.live_features import extended_space
        super().__init__()
        self.base = ObservedContract(include_grid=include_grid)
        self.observation_space = extended_space(self.base.observation_space)
        self.action_space = gym.spaces.MultiDiscrete([9, 2, 2, 2])

    def encode(self, raw):
        from touhou_ai.live_features import encode_extended
        raw = scoped(raw)
        base = self.base.encode(raw | {'lasers': []})
        # Preserve the first five channels used by the previous model for each laser.
        x, y = raw['player']['position']
        ordered = sorted(raw.get('lasers') or [], key=lambda e: ((e['position'][0]-x)**2+(e['position'][1]-y)**2,
                                                               *e['position'], *e['velocity_raw']))
        base['player'][12] = raw.get('lasers') is not None
        for index, entity in enumerate(ordered[:64]):
            ex, ey = entity['position']
            vx, vy = entity['velocity_raw']
            base['lasers'][index] = [1, (ex-x)/384, (ey-y)/448, vx/10, vy/10]
        return encode_extended(raw, base)


class FocusedObservedContract(ExtendedObservedContract):
    def __init__(self):
        super().__init__(include_grid=False)
        self.observation_space.spaces['near_bullets'] = gym.spaces.Box(-1, 1, (10, 8), dtype=np.float32)
        self.observation_space.spaces['previous_rewards'] = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.observation_space.spaces['bomb_clock'] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)

    def encode(self, raw, previous_rewards=None):
        from touhou_ai.focused_policy import reward_input
        observation = super().encode(raw)
        observation['near_bullets'] = observation['bullets'][:10].copy()
        observation['previous_rewards'] = reward_input(previous_rewards)
        observation['bomb_clock'] = np.zeros(1, dtype=np.float32)
        return observation


def hit_events(before, after, allow_stage_transition=False):
    stages = (before['stage'], before['stage']+1) if allow_stage_transition else (before['stage'],)
    if (after["stage"] not in stages or after['stage'] not in range(1, 7)
            or after["replay_mode"] != 0 or after["mode_flags"] not in (0, 4)):
        raise ValueError("unvalidated stage/menu transition")
    difference = before["lives_raw"]-after["lives_raw"]
    if difference not in (-1, 0, 1):
        raise ValueError("unexpected life change; do not infer a reward")
    return ([{"id": f"life-loss:{after['stage']}:{after['stage_frame']}", "kind": "hit", "confirmed": True}]
            if difference == 1 else [])


def fingerprint(model):
    digest = hashlib.sha256()
    for parameter in model.policy.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def resume_manifest(checkpoint, extended=False, dual_grid=False):
    from touhou_ai.live_features import EXTENDED_CONTRACT
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("resume checkpoint must be a managed local artifact")
    manifest = json.loads((checkpoint.parent / "status.json").read_text(encoding="utf-8"))
    from touhou_ai.focused_policy import CONTRACT as FOCUSED_CONTRACT
    from touhou_ai.dual_grid import CONTRACT as GRID_CONTRACT
    from touhou_ai.live_action_grid import CONTRACT as ACTION_GRID_CONTRACT
    accepted = (GRID_CONTRACT,ACTION_GRID_CONTRACT) if dual_grid else ((FOCUSED_CONTRACT,) if extended else (CONTRACT,))
    if manifest.get("backend") != "real_th10" or manifest.get("contract") not in accepted:
        raise ValueError("not a compatible real-game checkpoint")
    matches = [e for e in manifest["episodes"] if e["checkpoint"] == checkpoint.name]
    if len(matches) != 1 or not matches[0].get("reload_verified"):
        raise ValueError("checkpoint is not verified by this run")
    expected = matches[0].get("checkpoint_sha256")
    if not expected or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("checkpoint integrity mismatch")
    return manifest, matches[0]


def migrate_extended(old, new):
    """Keep existing features/heads; new inputs start at zero influence. Optimizer resets."""
    source = old.policy.state_dict()
    target = new.policy.state_dict()
    for key, value in target.items():
        previous = source[key]
        if previous.shape == value.shape:
            value.copy_(previous)
        elif key.startswith('action_net.'):
            value.zero_()
            value[:13].copy_(previous[:13])
            if key.endswith('bias'):
                value[-1] = -4
        elif value.ndim == previous.ndim == 2 and value.shape[0] == previous.shape[0] and value.shape[1] > previous.shape[1]:
            value.zero_()
            value[:, :previous.shape[1]].copy_(previous)
        else:
            raise ValueError(f'unsupported migration: {key}')
    new.policy.load_state_dict(target)
    new.num_timesteps = old.num_timesteps
    new._n_updates = old._n_updates
    return new


def finish_buffer(buffer, count, last_values, terminal):
    if count < 2:
        raise ValueError("not enough real transitions to update")
    # PPO uses the collected episode, never zero-filled padding or invented transitions.
    buffer.observations = {k: v[:count] for k, v in buffer.observations.items()}
    for name in ("actions", "rewards", "episode_starts", "values", "log_probs", "advantages", "returns"):
        setattr(buffer, name, getattr(buffer, name)[:count])
    buffer.buffer_size = buffer.pos = count
    buffer.full = True
    buffer.compute_returns_and_advantage(last_values, np.asarray([terminal]))


def game_command(action, recovery=False):
    executable = shutil.which("pwsh")
    if not executable:
        raise RuntimeError("PowerShell 7 is required")
    result = subprocess.run([executable, "-NoProfile", "-File", str(ROOT / "scripts/game.ps1"), action]
                            + (["-RecoveryStop"] if recovery else []),
                            cwd=ROOT, timeout=90, capture_output=True, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"game {action} failed: {result.stdout} {result.stderr}")
    print(result.stdout.strip(), flush=True)


def train(output, episodes=3, max_steps=1800, max_seconds=600, resume=None, continue_managed=False, extended=False, resume_paused=False, continuous=False, dual_grid=False, directml=False, detailed_logs=False, evasion_only=False, transfer_evasion=None, upgrade_progress_power=False, shot_study_games=0, shot_reward_scale=1., add_power_reward=False, upgrade_jitter=False, update_shot_reward=False):
    from touhou_ai.live_features import EXTENDED_CONTRACT, bomb_events
    from touhou_ai.focused_policy import CONTRACT as FOCUSED_CONTRACT, FocusedFeatures
    from touhou_ai.dual_grid import (CONTRACT as GRID_CONTRACT, SPEC as GRID_SPEC,
                                    DualGridContract, DualGridFeatures, GridRolloutBuffer)
    from touhou_ai.live_rewards import VERSION as REWARD_VERSION, WEIGHTS as REWARD_WEIGHTS, ENABLED as REWARD_ENABLED
    from touhou_ai.evasion_only import EvasionPolicy, DeathOnlyRewards, ACTION_CONTRACT
    if evasion_only:
        if not dual_grid:
            raise ValueError('evasion-only requires dual-grid')
        from touhou_ai.evasion_only import VERSION as REWARD_VERSION, WEIGHTS as REWARD_WEIGHTS, ENABLED as REWARD_ENABLED
    extended = extended or dual_grid
    if directml and not dual_grid:
        raise ValueError('DirectML updates require the tested dual-grid policy')
    directml_enabled = directml
    profile = json.loads((ROOT / "configs/sharu-inspired-v1.json").read_text(encoding="utf-8"))
    from touhou_ai.live_model import model_config, algorithm_class
    selection = model_config(profile, evasion_only, dual_grid)
    action_grid = selection['cnn_architecture'] == 'narrow_action_grid'
    from touhou_ai.live_action_grid import (CONTRACT as ACTION_GRID_CONTRACT, SPEC as ACTION_GRID_SPEC,
                                           LiveActionGridContract, build_live_action_model)
    contract_id = ACTION_GRID_CONTRACT if action_grid else (GRID_CONTRACT if dual_grid else (FOCUSED_CONTRACT if extended else CONTRACT))
    observation_scope = ACTION_GRID_SPEC if action_grid else (GRID_SPEC if dual_grid else BULLET_SCOPE)
    if resume_paused and not continue_managed:
        raise ValueError('resume-paused requires continue-managed')
    if not 1 <= episodes <= 5 or not 32 <= max_steps <= (18000 if continuous else 2400) or not 30 <= max_seconds <= 900:
        raise ValueError("bounded rehearsal budgets required")
    prior = resume_manifest(resume, extended, dual_grid) if resume is not None else None
    if update_shot_reward:
        from touhou_ai.live_rewards import validate_shot_update
        if not prior or evasion_only or not dual_grid or transfer_evasion or upgrade_progress_power or add_power_reward or upgrade_jitter or shot_study_games:
            raise ValueError('Shot update requires a full v25 resume')
        validate_shot_update(prior[0])
    if upgrade_jitter:
        raise ValueError('Vibration reward retired; start a fresh v25 campaign')
    if add_power_reward:
        from touhou_ai.live_rewards import validate_power_gain_upgrade
        if not prior or evasion_only or transfer_evasion or not dual_grid or upgrade_progress_power:
            raise ValueError('Power reward addition requires a full v19 dual-grid resume')
        validate_power_gain_upgrade(prior[0])
    if upgrade_progress_power:
        from touhou_ai.live_rewards import validate_power_upgrade
        if not prior or evasion_only or transfer_evasion or not dual_grid:
            raise ValueError('Power upgrade requires a full dual-grid resume')
        validate_power_upgrade(prior[0])
    if prior and not (upgrade_progress_power or add_power_reward or upgrade_jitter or update_shot_reward) and prior[0].get('reward_version') != REWARD_VERSION:
        raise ValueError('old reward checkpoint cannot resume the reset learning campaign')
    if prior and prior[0].get('bullet_scope') != observation_scope:
        raise ValueError('checkpoint bullet observation scope differs')
    if prior and extended and not (upgrade_progress_power or add_power_reward or upgrade_jitter or update_shot_reward) and prior[0].get('reward_weights') != REWARD_WEIGHTS:
        raise ValueError('reward weights changed; explicit new campaign required')
    if shot_study_games:
        if (not 1 <= shot_study_games <= 5 or shot_reward_scale not in (1., .5)
                or not prior or not dual_grid or evasion_only or not continuous or upgrade_progress_power or add_power_reward):
            raise ValueError('shot study requires 1-5 full continuous games from the standard checkpoint')
        REWARD_WEIGHTS = dict(REWARD_WEIGHTS, damage=REWARD_WEIGHTS['damage']*shot_reward_scale)
        if shot_reward_scale != 1.:
            REWARD_VERSION += '-shot-half-study'
    elif shot_reward_scale != 1.:
        raise ValueError('shot scaling requires a bounded study')
    if prior and extended and prior[0].get('bomb_schedule') != BOMB_SCHEDULE:
        raise ValueError('bomb schedule changed; explicit new campaign required')
    settings = dict(profile['provisional_ppo'])
    if evasion_only or dual_grid:
        settings.update(profile.get('evasion_ppo_overrides' if evasion_only else 'full_ppo_overrides', {}))
    if prior and prior[0].get('configured_ppo') != settings:
        raise ValueError('PPO settings changed; start a fresh campaign')
    policy_overrides = profile.get('evasion_policy_overrides' if evasion_only else 'full_policy_overrides', {})
    if not evasion_only and not dual_grid:
        policy_overrides = {}
    share_features = policy_overrides.get('share_features_extractor', True)
    from touhou_ai.live_model import model_config, algorithm_class
    selection = model_config(profile, evasion_only, dual_grid, prior[0] if prior else None)
    algorithm = algorithm_class(selection['algorithm'])
    from touhou_ai.narrow_grid import NarrowGridFeatures
    grid_features = NarrowGridFeatures if selection['cnn_architecture'] == 'narrow_grid' else DualGridFeatures
    if prior and prior[0].get('share_features_extractor', True) != share_features:
        raise ValueError('Feature sharing changed; start a fresh campaign')
    verify_settings(settings, prior[0] if prior else None)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    contract = LiveActionGridContract() if action_grid else (DualGridContract() if dual_grid else (FocusedObservedContract() if extended else ObservedContract()))
    model = build_live_action_model(settings, evasion_only=evasion_only) if action_grid else algorithm(EvasionPolicy if evasion_only else (TimedBombPolicy if extended else "MultiInputPolicy"), contract, device="cpu", **settings,
                rollout_buffer_class=GridRolloutBuffer if dual_grid else DictRolloutBuffer,
                policy_kwargs={"share_features_extractor": share_features, "features_extractor_class": grid_features if dual_grid else (FocusedFeatures if extended else NumericalFeatures),
                               "net_arch": {"pi": [256, 128] if extended else [128, 128],
                                            "vf": [256, 128] if extended else [128, 128]}})
    if prior is not None:
        if prior[0]['contract'] != contract_id:
            from touhou_ai.checkpoint_rng import load_preserving_rng
            model = migrate_extended(load_preserving_rng(PPO, resume, device='cpu'), model)
        else:
            from touhou_ai.checkpoint_rng import load_preserving_rng
            model = load_preserving_rng(algorithm, resume, env=contract, device="cpu")
        if getattr(model, 'collector_rng_state', None) is not None:
            from touhou_ai.checkpoint_rng import restore_rng
            restore_rng(model.collector_rng_state)
        if model.num_timesteps != prior[1]["total_steps"]:
            raise ValueError("checkpoint timestep metadata mismatch")
        for key, value in settings.items():
            actual = getattr(model, key)
            if callable(actual):
                actual = actual(1.0)
            if actual != value:
                raise ValueError(f"resume PPO configuration differs: {key}")
    transfer_info = None
    if transfer_evasion is not None:
        if resume is not None or evasion_only or not action_grid:
            raise ValueError('Transfer requires a new full action-grid campaign')
        source_manifest, source_episode = resume_manifest(transfer_evasion, True, True)
        if source_manifest.get('bullet_scope') != observation_scope:
            raise ValueError('Transfer observation scope mismatch')
        from touhou_ai.full_transfer import transfer_evasion as transfer_weights
        transfer_weights(transfer_evasion, source_manifest, model)
        transfer_info = dict(checkpoint=str(transfer_evasion),
                             sha256=source_episode['checkpoint_sha256'],
                             source_updates=source_episode['total_updates'],
                             value_scale=1/60, optimizer_reset=True)
    report = {"status": "running", "backend": "real_th10", "contract": contract_id,
              "episodes": [], "gameplay_training_steps": model.num_timesteps,
              "updates": prior[1]["total_updates"] if prior else 0,
              "full_profile_ready": False, "disabled": ["bomb", "damage_reward", "kill_reward",
              "stage_clear_reward", "stable_ids", "acceleration", "laser_geometry"],
              "configured_ppo": settings,
              "effective_ppo": settings | {"n_steps": "variable_episode_length"}, "max_seconds": max_seconds,
              "resumed_from": str(resume) if resume is not None else None,
              "progression": [{"stage": "input_and_update_connected", "enabled": ["Normal", "Reimu B",
                  "2 frames/action", "9 directions", "shoot", "focus", "hit=-5", "original PPO parameters"]}],
              "update_boundary": "game_over_menu", "telemetry_drops": 0,
              "between_episodes": "stage1_continue; stage2_plus_title_then_new_stage1", "phase": "initializing",
              "first_episode_started_midplay": resume_paused,
              "cpu_threads": {"inference": 1, "optimization": 8},
              "optimization_backend": "directml" if directml else "cpu"}
    report.update(selection)
    report['transfer_evasion'] = transfer_info
    report['rng_contract'] = 'continuous-checkpoint-and-worker-v1'
    report['resume_rng_restored'] = bool(prior and getattr(model, 'collector_rng_state', None) is not None)
    if action_grid:
        report['action_risk_prediction'] = ACTION_GRID_SPEC['action_grid']
        report['initialization'] = 'evasion weights transferred; value output /60; fresh optimizer' if transfer_info else 'fresh seed; baseline tensors copied and added merge connections zero; no checkpoint warm start'
    report['action_contract'] = ACTION_CONTRACT if evasion_only else 'standard-v1'
    report['evasion_only'] = evasion_only
    report['share_features_extractor'] = model.policy.share_features_extractor
    report['stage_transitions'] = []
    report['ui_stats_transport'] = 'shared_memory_v1'
    report['detailed_logs'] = detailed_logs
    ui_stats = UiStats(output)
    report['recoveries'] = []
    report['reward_version'] = REWARD_VERSION
    report['jitter_detector'] = JITTER_SPEC if extended and not evasion_only and 'jitter' in REWARD_ENABLED else None
    report['discount_contract'] = DISCOUNT_CONTRACT
    report['reward_weights'] = REWARD_WEIGHTS
    if shot_study_games:
        report['shot_reward_study'] = {'games': shot_study_games, 'scale': shot_reward_scale,
            'common_checkpoint': str(resume), 'source_reward_version': prior[0]['reward_version']}
    if upgrade_progress_power or add_power_reward or upgrade_jitter or update_shot_reward:
        report['reward_transition'] = {'from': prior[0]['reward_version'], 'to': REWARD_VERSION,
            'old_weights': prior[0]['reward_weights'], 'new_weights': REWARD_WEIGHTS,
            'policy_optimizer_rng_preserved': True}
    report['reward_enabled'] = REWARD_ENABLED
    report['bomb_schedule'] = BOMB_SCHEDULE if extended else None
    report['progress_validation'] = ('enabled: verified_ecl_progress_v1; stage 1 live-validated; '
                                     'stages 2-6 script-checked, full live validation pending; no inferred awards')
    report['bullet_scope'] = observation_scope
    if dual_grid:
        report['geometry_validation'] = 'binary-derived AABB; live bullet hit/miss trace pending'
        report['grid_rollout_bytes'] = max_steps * sum(
            int(np.prod(s.shape)) * (2 if k.endswith('_grid') else 4)
            for k, s in contract.observation_space.spaces.items())
    report['disabled'] = ['stable_ids'] + (['shoot', 'bomb', 'damage_reward', 'progress_reward'] if evasion_only else [])
    if evasion_only:
        report['progression'][0]['enabled'] = ['Normal', 'Reimu B', '2 frames/action', '9 directions', 'focus', 'hit=-60', 'fresh evasion experiment']
    if not evasion_only:
        report['progression'][0]['enabled'] = ['Normal', 'Reimu B', '2 frames/action', '9 directions', 'focus', 'shoot', 'scheduled bomb', 'rewards-v17']
    report['combat_validation_pending'] = ['milestone_arrival_defeat_timeout_identity', 'post_bomb_residual_damage']
    if extended:
        report['disabled'] += ['laser_field_validation']
        report['extended_observations'] = ['bomb_state', 'enemy_hp_ratio', 'boss_flag', 'spell_id',
                                            'backward_acceleration', 'binary_derived_laser_rectangle']
        report['optimizer_reset_for_migration'] = bool(prior and prior[0]['contract'] != contract_id)
    if dual_grid:
        report['extended_observations'] = ['local_collision_and_bullet_offset_grid',
            'global_bullets_enemies_items_player_lasers_grid', 'live_player_shot_coverage', 'player_state', 'previous_rewards']
    atomic_json(output / "status.json", report)
    report['continuous'] = continuous
    report['recovery_attempt_limit'] = None if continuous else 2
    if continuous:
        report['max_seconds'] = None
    deadline = float('inf') if continuous else time.monotonic()+max_seconds
    initial_hash = fingerprint(model)
    runtime = hold = None
    fresh_ready = False

    def interrupted():
        return (output / 'STOP').exists() or time.monotonic() >= deadline

    def restart_stage_one():
        nonlocal runtime, fresh_ready
        if interrupted():
            raise CollectionInterrupted('stop before recovery restart')
        report['phase'] = 'restarting_stage_one'
        atomic_json(output / 'status.json', report)
        game_command('start')
        record = json.loads((ROOT / '.runtime/game.json').read_text(encoding='utf-8-sig'))
        runtime = LiveRuntime(record['Id'])
        report['game_pid'] = record['Id']
        if interrupted():
            raise CollectionInterrupted('stop during recovery startup')
        start_episode(runtime)
        state = runtime.snapshot(full=False)
        if state['stage'] != 1:
            raise RuntimeError('recovery did not reach stage one')
        fresh_ready = True
        report['recoveries'][-1].update(restarted=True, new_pid=record['Id'], stage=1)

    try:
        import itertools
        import shutil
        for episode in (itertools.count() if continuous else range(episodes)):
            if shot_study_games and episode >= shot_study_games:
                report.update(status='stopped', stop_reason='shot_study_complete')
                break
            if continuous and shutil.disk_usage(output).free < 5 * 1024**3:
                report.update(status='stopped', stop_reason='disk_free_below_5_GiB')
                break
            if (output / "STOP").exists() or time.monotonic() >= deadline:
                report["status"] = "stopped"
                break
            budget = max_steps
            buffer_type = GridRolloutBuffer if dual_grid else DictRolloutBuffer
            buffer = buffer_type(budget, contract.observation_space, contract.action_space,
                                      device="cpu", gamma=settings["gamma"], gae_lambda=settings["gae_lambda"])
            rewards = DeathOnlyRewards() if evasion_only else (LiveRewards(weights=REWARD_WEIGHTS) if extended else EventRewards())
            episode_id = f"{output.name}-{episode+1}"
            rewards.reset(episode_id)
            jitter = VibrationReward()
            total_reward, count, hits = 0.0, 0, 0
            jitter_count, jitter_total = 0, 0.
            power_gain_raw, power_gain_reward = 0, 0.
            study_metrics = dict(shot_damage_hp=0., shot_reward=0., shoot_decisions=0, first_hit_decision=None)
            ui_stats.begin_episode()
            terminal = False
            committed_terminal = False
            recovery = None
            stop_requested = False
            log_path = output / f"episode-{episode+1}.jsonl"
            try:
                if runtime is None:
                    # An existing session is accepted only with an explicit switch and a verified game over.
                    if not continue_managed:
                        game_command("start")
                    record = json.loads((ROOT / ".runtime/game.json").read_text(encoding="utf-8-sig"))
                    runtime = LiveRuntime(record["Id"])
                    report["game_pid"] = record["Id"]
                    if continue_managed:
                        if resume_paused:
                            from touhou_ai.live_reset import resume_paused_episode
                            resume_paused_episode(runtime)
                        else:
                            continue_episode(runtime)
                    else:
                        start_episode(runtime)
                elif not fresh_ready:
                    continue_episode(runtime)
                fresh_ready = False
                report["phase"] = "playing"
                atomic_json(output / "status.json", report)
                before = runtime.snapshot()
                atomic_json(output / f"initial-state-{episode+1}.json", before)
                def interrupted():
                    return (output / 'STOP').exists() or time.monotonic() >= deadline

                def publish_transition(state, transition):
                    waiting = packet(state, episode_id, 'live')
                    waiting['learning'] = {'steps': model.num_timesteps, 'updates': report['updates'],
                        'contract': contract_id, 'phase': 'stage_transition', 'transition': transition}
                    publish_telemetry(waiting)

                before, _ = bridge_transition(runtime, before, before['stage'],
                    should_stop=interrupted, publish=publish_transition)
                observation = contract.encode(before)
                bomb_clock = BombClock()
                if extended:
                    observation['bomb_clock'] = bomb_clock.encode()
                previous_timing = None
                with (log_path.open("x", encoding="utf-8") if detailed_logs else nullcontext(None)) as log:
                    for step in range(budget):
                        if (output / "STOP").exists() or time.monotonic() >= deadline:
                            stop_requested = True
                            break
                        cycle_started = time.perf_counter()
                        with torch.no_grad():
                            tensor, _ = model.policy.obs_to_tensor(observation)
                            if isinstance(model.policy, TimedBombPolicy):
                                action, values, log_prob, distributions = model.policy.forward_with_distribution(tensor)
                            else:
                                action, values, log_prob = model.policy(tensor)
                                distributions = model.policy.get_distribution(tensor).distribution
                            probabilities = [d.probs[0].cpu().tolist() for d in distributions]
                        inference_ms = (time.perf_counter()-cycle_started)*1000
                        step_started = time.perf_counter()
                        chosen = action[0].cpu().numpy()
                        if evasion_only and (chosen[1] != 0 or chosen[3] != 0):
                            raise RuntimeError('disabled evasion action sampled')
                        after = runtime.step_gameplay(input_mask(chosen.tolist()), 2)
                        applied = after["input_state_raw"][0]
                        if applied != input_mask(chosen.tolist()) and not (after['lives_raw'] < 0 and applied == 0):
                            raise RuntimeError("actual input differs from sampled policy action")
                        action_result = after
                        # Preserve action events: transition/menu stepping clears gate batches.
                        action_events = list(after.get('combat_reward_events', []))
                        try:
                            after, transition = bridge_transition(runtime, after, before['stage'],
                                should_stop=interrupted, publish=publish_transition)
                        except CollectionInterrupted:
                            stop_requested = True
                            break
                        after['sample_ms'] = (time.perf_counter()-step_started)*1000
                        after['combat_reward_events'] = action_events
                        events = observed_events(before, after, chosen.tolist())
                        if before['stage'] != after['stage']:
                            report['stage_transitions'].append(transition | {'episode': episode+1,
                                'updates': report['updates'], 'episode_continues': True})
                            atomic_json(output / 'status.json', report)
                        jitter_fraction = jitter.observe(before, after) if extended and not evasion_only and 'jitter' in REWARD_ENABLED else 0.
                        if jitter_fraction:
                            events.append(dict(id=f"jitter:{after['stage']}:{after['stage_frame']}",
                                kind='jitter', confirmed=True, source=JITTER_SOURCE, amount=jitter_fraction))
                        reward, components = rewards.calculate(episode_id, events)
                        if shot_study_games:
                            study_metrics['shot_damage_hp'] += sum(e['amount'] for e in events if e['kind']=='damage' and e.get('bomb_state')==0)
                            study_metrics['shot_reward'] += components.get('damage', 0.)
                            study_metrics['shoot_decisions'] += int(chosen[1])
                            if study_metrics['first_hit_decision'] is None and any(e['kind']=='hit' for e in events):
                                study_metrics['first_hit_decision'] = count+1
                        terminal = after["lives_raw"] < 0
                        if after["player"] is None and not terminal:
                            raise RuntimeError("player disappeared outside verified terminal")
                        encode_started = time.perf_counter()
                        next_observation = observation if after["player"] is None else (
                            contract.encode(after, components) if extended else contract.encode(after))
                        if extended:
                            bomb_clock.advance(2)
                            next_observation = dict(next_observation)
                            next_observation['bomb_clock'] = bomb_clock.encode()
                        encode_ms = (time.perf_counter()-encode_started)*1000
                        buffer.add(observation, chosen.reshape(1, -1), np.array([reward]),
                                   np.array([step == 0]), values, log_prob)
                        model.num_timesteps += 1
                        count += 1
                        committed_terminal = terminal
                        last_observation = next_observation
                        hits += sum(event['kind'] == 'hit' for event in events)
                        total_reward += reward
                        jitter_count += sum(event['kind'] == 'jitter' for event in events)
                        jitter_total += components.get('jitter', 0.)
                        power_gain_raw += sum(e['amount_raw'] for e in events if e['kind'] == 'power_gain')
                        power_gain_reward += components.get('power_gain', 0.)
                        telemetry_started = time.perf_counter()
                        telemetry = packet(after, episode_id, "live")
                        if dual_grid:
                            # Show the complete scene, not the previous top-40 selection.
                            telemetry['ai_observation'] = {
                                'bullet_scope': observation_scope,
                                'bullets': telemetry['entities']['bullets'],
                                'items': telemetry['entities']['items'], 'bullet_grid': [],
                                'local_viewport': {'center': telemetry['player']['position'] if telemetry['player'] else None,
                                                   'pixels': 192, 'cell_pixels': 2},
                                'display': 'raw_scene_with_grid_extent', 'frame': after['stage_frame']}
                        from touhou_ai.model_monitor import model_metadata
                        if action_grid:
                            telemetry['ai_observation']['action_grid'] = observation['action_grid'].tolist()
                            telemetry['ai_observation']['action_grid_frame'] = before['stage_frame']
                        telemetry['model'] = model_metadata(model, contract_id,
                            str(resume) if resume else None,
                            reward_weights=REWARD_WEIGHTS)
                        telemetry['model']['reward_enabled'] = REWARD_ENABLED
                        telemetry['model']['reward_version'] = REWARD_VERSION
                        if extended:
                            telemetry['ai_observation']['previous_rewards'] = observation['previous_rewards'].tolist()
                        telemetry["policy"] = {"source": "real_th10", "trained_updates": report["updates"],
                            "value": float(values.item()), "action": chosen.tolist(),
                            "directions": probabilities[0], "shoot": probabilities[1][1],
                            "focus": probabilities[2][1], "bomb": probabilities[3][1] if extended else 0.0,
                            "observation_frame": before["stage_frame"]}
                        if extended:
                            telemetry['policy'].update(bomb_decision=bool(observation['bomb_clock'][0] == 0),
                                bomb_clock=observation['bomb_clock'].tolist(), bomb_schedule=BOMB_SCHEDULE)
                        telemetry["reward"] = {"total": reward, "episode_return": total_reward,
                            "components": components, "enabled": REWARD_ENABLED}
                        telemetry["learning"] = {"steps": model.num_timesteps, "updates": report["updates"],
                            "contract": contract_id, "partial_profile": True, "phase": "playing"}
                        telemetry["capabilities"].update(policy_connected=True, live_training=True)
                        timing = {"inference_ms": inference_ms, "step_observation_transition_ms": after["sample_ms"],
                                  "encode_ms": encode_ms, "telemetry_ms": (time.perf_counter()-telemetry_started)*1000}
                        timing.update(action_result.get("runtime_timing_ms", {}))
                        # Completed previous cycle, including serialization/publication.
                        # Reuse the existing shared-memory packet; no per-step disk IO.
                        if previous_timing is not None:
                            telemetry['previous_cycle_timing'] = previous_timing
                        log_started = time.perf_counter()
                        if log is not None:
                            log.write(json.dumps({"raw": after, "telemetry": telemetry, "events": events,
                                                  "timing_ms": timing, "previous_step_timing_ms": previous_timing,
                                                  'transition': transition}, allow_nan=False)+"\n")
                            log.flush()
                        timing["log_ms"] = (time.perf_counter()-log_started)*1000
                        publish_started = time.perf_counter()
                        ui_stats.add(telemetry, events, terminal, episode_id, settings["gamma"])
                        timing['ui_stats_ms'] = (time.perf_counter()-publish_started)*1000
                        transport_started = time.perf_counter()
                        if not publish_telemetry(telemetry):
                            report["telemetry_drops"] += 1
                        timing['transport_ms'] = (time.perf_counter()-transport_started)*1000
                        timing["publish_ms"] = (time.perf_counter()-publish_started)*1000
                        timing["total_ms"] = (time.perf_counter()-cycle_started)*1000
                        previous_timing = dict(timing, step=count, stage=after['stage'],
                            frame=after['stage_frame'], terminal=terminal,
                            transition=action_result.get('transition'),
                            bullets=len(after.get('bullets') or []), lasers=len(after.get('lasers') or []),
                            active_lasers=sum(bool(l.get('collision') and l['collision']['active']) for l in (after.get('lasers') or [])),
                            boss=any(e.get('is_boss') for e in (after.get('enemies') or [])))
                        observation, before = next_observation, after
                        if count % 120 == 0:
                            print(json.dumps({"episode": episode+1, "steps": count, "frame": after["stage_frame"],
                                              "hits": hits, "return": total_reward}), flush=True)
                        if terminal:
                            break
                ui_stats.flush()
                if previous_timing is not None:
                    atomic_json(output / "last-step-timing.json", previous_timing)
                with torch.no_grad():
                    last_values = model.policy.predict_values(model.policy.obs_to_tensor(observation)[0])
                # Release the last policy action before any optimizer work.
                if runtime is not None and terminal:
                    wait_game_over(runtime)
            except Exception as error:
                from touhou_ai.live_recovery import recovery_allowed
                if not recovery_allowed(runtime, error, interrupted(), len(report['recoveries']),
                                        max_attempts=None if continuous else 2):
                    raise
                recovery = {'error': str(error), 'valid_steps': count, 'restarted': False,
                            'old_pid': report['game_pid'], 'checkpoint': None}
                report['recoveries'].append(recovery)
                report['phase'] = 'recovering'
                atomic_json(output / 'status.json', report)
                try:
                    runtime.close()
                except Exception as close_error:
                    recovery['detach_error'] = str(close_error)
                runtime = None
                game_command('stop', recovery=True)
                if interrupted():
                    raise CollectionInterrupted('stop during recovery shutdown')
                terminal = committed_terminal
                if count < 32:
                    recovery['update_skipped'] = 'fewer than 32 validated transitions'
                    model.num_timesteps -= count
                    restart_stage_one()
                    atomic_json(output / 'status.json', report)
                    continue
                observation = last_observation
                with torch.no_grad():
                    last_values = model.policy.predict_values(model.policy.obs_to_tensor(observation)[0])
            if stop_requested or (not terminal and recovery is None):
                report["status"] = "stopped"
                report["stop_reason"] = "budget_or_stop_before_game_over; trajectory_not_updated"
                break
            report["phase"] = "optimizing_recovery" if recovery else "optimizing_at_game_over"
            atomic_json(output / "status.json", report)
            def publish_hold(state):
                waiting = packet(state, episode_id, "live")
                waiting["learning"] = {"steps": model.num_timesteps, "updates": report["updates"],
                    "contract": contract_id, "partial_profile": True, "phase": "optimizing_at_game_over"}
                waiting["reward"] = {"total": 0, "episode_return": total_reward,
                    "components": {}, "enabled": REWARD_ENABLED}
                if not publish_telemetry(waiting):
                    report["telemetry_drops"] += 1
            if recovery is None:
                hold = GameOverHold(runtime, publish_hold)
                hold.start()
            finish_buffer(buffer, count, last_values, terminal)
            model.rollout_buffer = buffer
            model.set_logger(configure(str(output / f"optimizer-{episode+1}"), ["json"]))
            prior_hash = fingerprint(model)
            # Keep single-observation inference lightweight; parallelize batch updates.
            torch.set_num_threads(8)
            optimization_started = time.monotonic()
            optimization_detail = {"backend": "cpu"}
            try:
                if directml_enabled:
                    from touhou_ai.directml_update import update
                    model, optimization_detail = update(model, buffer, output / f'directml-{episode+1}')
                    if optimization_detail['backend'] == 'cpu_fallback':
                        directml_enabled = False
                        report['optimization_backend'] = 'cpu_fallback'
                        report['directml_error'] = optimization_detail['error']
                else:
                    model.train()
            finally:
                optimization_seconds = time.monotonic() - optimization_started
                torch.set_num_threads(1)
            if fingerprint(model) == prior_hash or not all(torch.isfinite(p).all() for p in model.policy.parameters()):
                raise AssertionError("real PPO update failed")
            model.logger.dump(model.num_timesteps)
            model.logger.close()
            checkpoint = output / f"real-episode-{episode+1}.zip"
            from touhou_ai.checkpoint_rng import capture_rng
            model.collector_rng_state = capture_rng()
            model.save(checkpoint)
            expected = model.predict(observation, deterministic=True)[0]
            old_steps = model.num_timesteps
            from touhou_ai.checkpoint_rng import load_preserving_rng
            loaded = load_preserving_rng(algorithm, checkpoint, env=contract, device="cpu")
            np.testing.assert_array_equal(expected, loaded.predict(observation, deterministic=True)[0])
            if loaded.num_timesteps != old_steps or fingerprint(loaded) != fingerprint(model):
                raise AssertionError("checkpoint did not preserve real policy")
            if capture_rng() != model.collector_rng_state or loaded.collector_rng_state != model.collector_rng_state:
                raise AssertionError('checkpoint did not preserve collector randomness')
            model = loaded
            hold_samples = 0
            if hold is not None:
                hold_samples = hold.samples
                try:
                    hold.stop()
                except Exception as error:
                    from touhou_ai.live_recovery import recovery_allowed
                    if not recovery_allowed(runtime, error, interrupted(), len(report['recoveries']),
                                            max_attempts=None if continuous else 2):
                        raise
                    recovery = {'error': str(error), 'valid_steps': count, 'restarted': False,
                                'old_pid': report['game_pid'], 'checkpoint': checkpoint.name,
                                'fault_during_game_over_hold': True}
                    report['recoveries'].append(recovery)
                    try:
                        runtime.close()
                    except Exception as close_error:
                        recovery['detach_error'] = str(close_error)
                    runtime = None
                    game_command('stop', recovery=True)
            hold = None
            report["updates"] += 1
            report["gameplay_training_steps"] = model.num_timesteps
            report["episodes"].append({"episode": episode+1, "steps": count, "hits": hits,
                "return": total_reward, "terminated": terminal, "truncated": not terminal,
                "jitter_events": jitter_count, "jitter_reward": jitter_total,
                "power_acquired": power_gain_raw / 20, "power_gain_reward": power_gain_reward,
                "final_frame": before["stage_frame"], "budget": budget, "checkpoint": checkpoint.name,
                "parameters_changed": True, "reload_verified": True,
                "reload_rng_verified": True,
                "optimization_threads": 8, "optimization_seconds": optimization_seconds,
                "optimization": optimization_detail,
                "game_pid": report["game_pid"], "optimizer_menu_samples": hold_samples,
                "updated_at_game_over": recovery is None, "recovery_update": recovery is not None,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "total_steps": model.num_timesteps, "total_updates": report["updates"],
                "max_progress": ui_stats.best})
            if shot_study_games:
                report['episodes'][-1]['shot_study_metrics'] = study_metrics
            if terminal and recovery is None and not any(p["stage"] == "game_over_boundary_observed" for p in report["progression"]):
                report["progression"].append({"stage": "game_over_boundary_observed", "episode": episode+1,
                    "evidence": "reserve lives -1, game-over menu held with neutral input throughout PPO/save/reload"})
            # Publish checkpoint verification before any relaunch that might fail.
            if recovery is not None:
                recovery['checkpoint'] = checkpoint.name
                atomic_json(output / 'status.json', report)
                restart_stage_one()
            report["phase"] = "stage_one_ready" if recovery else "game_over_ready"
            final_telemetry = packet(runtime.snapshot(full=False), episode_id, "live")
            final_telemetry["learning"] = {"steps": model.num_timesteps, "updates": report["updates"],
                "contract": contract_id, "partial_profile": True, "phase": report['phase']}
            publish_telemetry(final_telemetry)
            atomic_json(output / "status.json", report)
            print(json.dumps(report["episodes"][-1]), flush=True)
        else:
            report["status"] = "passed"
        report["initial_policy_sha256"] = initial_hash
        report["final_policy_sha256"] = fingerprint(model)
    except CollectionInterrupted:
        report.update(status='stopped', stop_reason='stop_during_transition; trajectory_not_updated')
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        try:
            if hold is not None:
                hold.stop()
        finally:
            if runtime is not None:
                try:
                    state = runtime.snapshot(full=False)
                    if (state.get('lives_raw', -1) >= 0 and state.get('mode_flags') in (0, 4)
                            and state.get('player') is not None and state.get('pause_words', [0, -1])[1] == 0):
                        from touhou_ai.live_acceptance import pause
                        report['paused_on_exit'] = pause(runtime)['pause_words'][1] == 2
                except Exception as error:
                    report['pause_on_exit_error'] = str(error)
                finally:
                    runtime.close()
            report["game_left_open"] = runtime is not None
            atomic_json(output / "status.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--update-shot-reward", action="store_true")
    parser.add_argument("--upgrade-jitter", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--transfer-evasion", type=Path)
    parser.add_argument("--add-power-reward", action="store_true")
    parser.add_argument("--upgrade-progress-power", action="store_true")
    parser.add_argument("--shot-study-games", type=int, default=0)
    parser.add_argument("--shot-reward-scale", type=float, default=1.)
    parser.add_argument("--continue-managed", action="store_true")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--dual-grid", action="store_true", help="new dual-grid campaign; implies extended actions/rewards")
    parser.add_argument("--resume-paused", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--detailed-logs", action="store_true", help="opt-in per-step full observation diagnostic JSONL")
    parser.add_argument("--evasion-only", action="store_true", help="death-only rewards; shot and bomb disabled")
    parser.add_argument("--directml", action="store_true", help="isolated GPU updates with CPU rollback")
    args = parser.parse_args()
    try:
        train(args.output, args.episodes, args.max_steps, args.max_seconds, args.resume, args.continue_managed, args.extended, args.resume_paused, args.continuous, args.dual_grid, args.directml, args.detailed_logs, args.evasion_only, args.transfer_evasion, args.upgrade_progress_power, args.shot_study_games, args.shot_reward_scale, args.add_power_reward, args.upgrade_jitter, args.update_shot_reward)
    except BaseException as error:
        status_path = args.output / "status.json"
        if args.output.exists() and not status_path.exists():
            atomic_json(status_path, {"status": "failed", "backend": "real_th10", "contract": CONTRACT,
                "error": str(error), "episodes": [], "gameplay_training_steps": 0, "updates": 0,
                "resumed_from": str(args.resume) if args.resume is not None else None})
        raise
