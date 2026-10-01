import copy
import struct
import unittest
import numpy as np
from test_live import Memory
from test_dual_grid import state
from touhou_ai.th10_reader import Th10Reader
from touhou_ai.dual_grid import DualGridContract, GLOBAL_CHANNELS


class PlayerShotTests(unittest.TestCase):
    def fixture(self):
        memory = Memory()
        memory.put_pointer(0x477834, 0x1000000)
        data = bytearray(0x4478-0x3c0)
        memory.regions[0x10003c0] = data
        offset = 0x49c-0x3c0
        struct.pack_into('<ff', data, offset+0x14, 0, 100)
        struct.pack_into('<i', data, offset+0x40, 1)
        struct.pack_into('<I', data, offset+0x58, 0x2000000)
        descriptor = bytearray(0x34)
        struct.pack_into('<ff', descriptor, 0xc, 16, 48)
        memory.regions[0x2000000] = descriptor
        return memory, data, offset, descriptor

    def test_live_row_and_full_snapshot(self):
        memory, data, offset, descriptor = self.fixture()
        shots = Th10Reader(memory).snapshot()['player_shots']
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0]['hitbox_raw'], [16,48])
        self.assertEqual(shots[0]['position'], [0,100])
        self.assertIsNone(Th10Reader(memory).snapshot(False)['player_shots'])
        self.assertIsNone(Th10Reader(Memory()).snapshot()['player_shots'])
        for inactive in (0,2):
            struct.pack_into('<i', data, offset+0x40, inactive)
            struct.pack_into('<I', data, offset+0x58, 0)
            self.assertEqual(Th10Reader(memory).snapshot()['player_shots'], [])

    def test_native_top_edge_and_unknown_callback(self):
        memory, data, offset, descriptor = self.fixture()
        struct.pack_into('<f', data, offset+0x18, 23)
        self.assertEqual(Th10Reader(memory).snapshot()['player_shots'], [])
        descriptor[0x1d] = 3
        self.assertEqual(len(Th10Reader(memory).snapshot()['player_shots']), 1)
        struct.pack_into('<I', descriptor, 0x30, 0x401000)
        with self.assertRaisesRegex(ValueError, 'callback'):
            Th10Reader(memory).snapshot()

    def test_last_slot_and_invalid_values(self):
        memory, data, offset, descriptor = self.fixture()
        data[offset+127*0x5c:offset+128*0x5c] = data[offset:offset+0x5c]
        self.assertEqual(len(Th10Reader(memory).snapshot()['player_shots']), 2)
        struct.pack_into('<f', descriptor, 0xc, -1)
        with self.assertRaisesRegex(ValueError, 'hitbox'):
            Th10Reader(memory).snapshot()

    def test_grid_fraction_overlap_viewport_and_missing_data(self):
        env = DualGridContract(); raw = state()
        shot = {'position': [0,100], 'hitbox_raw': [16,48],
                'geometry': 'th10-player-shot-aabb-v1'}
        raw['player_shots'] = [shot, copy.deepcopy(shot)]
        obs = env.encode(raw)
        channel = GLOBAL_CHANNELS.index('player_shot_coverage')
        self.assertEqual(obs['global_grid'].shape, (12,56,48))
        self.assertEqual(obs['global_grid'][channel].sum(), 12)
        self.assertEqual(obs['previous_rewards'].sum(), 0)
        self.assertTrue(env.observation_space.contains(obs))
        raw['player_shots'] = [shot | {'position': [-196,100]}]
        self.assertEqual(env.encode(raw)['global_grid'][channel].sum(), 3)
        raw['player_shots'] = []
        self.assertEqual(env.encode(raw)['global_grid'][channel].sum(), 0)
        for missing in (None,):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                env.encode(raw | {'player_shots': missing})
        with self.assertRaisesRegex(ValueError, 'geometry'):
            env.encode(raw | {'player_shots': [shot | {'geometry': 'guess'}]})
