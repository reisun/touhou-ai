"""Bounded TH10 raw snapshots. Unknown fields stay unknown, never zero-filled."""
import math
import struct


class Th10Reader:
    def __init__(self, process):
        self.process = process
        self.previous_bullets = None

    def block(self, address, size):
        if not 0x10000 <= address < 0x80000000 or not 0 < size <= 0x500000:
            raise ValueError("invalid memory range")
        chunk = min(262144, getattr(self.process, "max_read_size", 16384))
        return b"".join(self.process.read(address + offset, min(chunk, size-offset))
                        for offset in range(0, size, chunk))

    def integer(self, address):
        return struct.unpack("<I", self.block(address, 4))[0]

    @staticmethod
    def floats(data, offset, count):
        values = struct.unpack_from("<" + "f" * count, data, offset)
        if any(not math.isfinite(v) or abs(v) > 1e7 for v in values):
            raise ValueError("invalid game float")
        return list(values)

    def globals(self):
        data = self.block(0x474C48, 0x5c)
        return {name: struct.unpack_from("<i", data, offset)[0] for name, offset in
                {"power_raw": 0, "character": 0x20, "shot": 0x24, "lives_raw": 0x28,
                 "difficulty": 0x2c, "stage": 0x34, "stage_frame": 0x40, "mode_flags": 0x58}.items()}

    def snapshot(self, full=True):
        result = self.globals()
        result['screen_state_raw'] = self.integer(0x491fb8)
        stage_manager = self.integer(0x477810)
        result['stage_manager_raw'] = ({'flags': self.integer(stage_manager + 0x58),
                                        'timer': self.integer(stage_manager + 0x14)}
                                       if stage_manager else None)
        # 0x418201 sets 0x800 during the incoming-stage animation;
        # 0x41840b..0x418471 clears it at timer 30 and resets the stage clock.
        result['stage_init_pending'] = (bool(result['stage_manager_raw']['flags'] & 0x800)
                                        if stage_manager else None)
        result["input_state_raw"] = struct.unpack("<5H", self.block(0x474e30, 10))
        replay = self.integer(0x477838)
        menu = self.integer(0x47784c)
        result["replay_mode"] = self.integer(replay+0x10) if replay else None
        result["menu_words"] = list(struct.unpack("<16i", self.block(menu, 64))) if menu else None
        pause = self.integer(0x477830)
        result["pause_words"] = list(struct.unpack("<12i", self.block(pause, 48))) if pause else None
        bomb = self.integer(0x4776ec)
        result["bomb"] = ({"state": self.integer(bomb+0x28),
                           "timer": self.integer(bomb+0x18)} if bomb else None)
        spell = self.integer(0x4776f4)
        result["spell"] = ({"id_raw": self.integer(spell+0x3788),
                            "flags_raw": self.integer(spell+0x378c)} if spell else None)
        gui = self.integer(0x47770c)
        result['dialogue_raw'] = self.integer(gui+0x9eb8) if gui else None
        player = self.integer(0x477834)
        result.update(player=None, bullets=None, items=None, enemies=None, lasers=None, player_shots=None,
                      stable_entity_ids=False, reward_events_verified=False)
        if player:
            data = self.block(player + 0x3c0, 0x4478-0x3c0)
            result["player"] = {"position": self.floats(data, 0, 2),
                                "velocity_raw": list(struct.unpack_from("<ii", data, 0x30)),
                                "hitbox_raw": self.floats(data, 0x5c, 2),
                                "status": struct.unpack_from("<i", data, 0x98)[0],
                                "invincibility_raw": struct.unpack_from("<i", data, 0x3f50)[0],
                                "focus_raw": struct.unpack_from("<i", data, 0x40b4)[0]}
        if full and player:
            result["player_shots"] = self.player_shots(data)
        if not full:
            return result
        for kind, pointer, start, stride, count, status_offset in (
                ("bullets", 0x4776f0, 0x60, 0x7f0, 2000, 0x446),
                ("items", 0x477818, 0x14, 0x3f0, 2198, 0x3dc)):
            base = self.integer(pointer)
            if not base:
                continue
            data = self.block(base+start, stride*count)
            result[kind] = []
            for index in range(count):
                offset = index*stride
                status = struct.unpack_from("<h" if kind == "bullets" else "<i", data, offset+status_offset)[0]
                if not status:
                    continue
                position_offset, velocity_offset = (0x3b4, 0x3c0) if kind == "bullets" else (0x3ac, 0x3b8)
                entity = {"slot": index, "status": status,
                          "position": self.floats(data, offset+position_offset, 2),
                          "velocity_raw": self.floats(data, offset+velocity_offset, 2)}
                if kind == "bullets":
                    entity["flags_raw"] = struct.unpack_from("<I", data, offset)[0]
                    entity["age_frames"] = struct.unpack_from("<i", data, offset+0x3fc)[0]
                    entity["hitbox_raw"] = self.floats(data, offset+0x3f0, 2)
                    entity["type"] = struct.unpack_from("<i", data, offset+0x460)[0]
                else:
                    entity["type"] = struct.unpack_from("<i", data, offset+0x3e0)[0]
                result[kind].append(entity)
            if kind == "bullets":
                self.add_acceleration(result, base)
        for kind, pointer, head_offset in (("enemies", 0x477704, 0x58), ("lasers", 0x47781c, 0x18)):
            base = self.integer(pointer)
            if not base:
                continue
            node = self.integer(base+head_offset)
            result[kind], visited = [], set()
            while node:
                if node in visited or len(visited) >= 4096:
                    raise ValueError("cyclic/oversized entity list")
                visited.add(node)
                if kind == "enemies":
                    raw, following = struct.unpack("<II", self.block(node, 8))
                    data = self.block(raw+0x1068, 0x2484-0x1068)
                    status = struct.unpack_from("<I", data, 0x1418)[0]
                    if not status & 0x52 or status & 0x8000:
                        result[kind].append({"address": raw, "position": self.floats(data, 0, 2),
                                             "velocity_raw": self.floats(data, 0xc, 2),
                                             "hitbox_raw": self.floats(data, 0x8c, 2),
                                             "hp": struct.unpack_from("<i", data, 0x1394)[0],
                                             "hp_max": struct.unpack_from("<i", data, 0x1398)[0],
                                             "flags_raw": status, "is_boss": bool(status & 0x8000)})
                else:
                    data = self.block(node, 0x48)
                    following = struct.unpack_from("<I", data, 8)[0]
                    vtable, = struct.unpack_from('<I', data)
                    laser_kind = {0x46da60: 'line', 0x46da10: 'infinite'}.get(vtable, 'unknown')
                    state, = struct.unpack_from('<i', data, 0xc)
                    position = self.floats(data, 0x24, 2)
                    geometry = self.floats(data, 0x3c, 3)
                    from touhou_ai.live_features import laser_collision
                    result[kind].append({"address": node, "position": position,
                                         "velocity_raw": self.floats(data, 0x30, 2),
                                         "kind": laser_kind, "state": state,
                                         "angle_length_width_raw": geometry,
                                         "collision": laser_collision(laser_kind, state, position, *geometry)})
                node = following
        return result

    def player_shots(self, player_data):
        """Pinned 0x428630 collision owner's 128 runtime rows, not visual sprites.

        player_data begins at player+0x3c0. Descriptor extents are full sizes.
        State 2 is a hit animation and no longer participates in collisions.
        Nonzero callbacks need their own geometry validation; never guess them.
        """
        shots, descriptors = [], {}
        for slot in range(128):
            offset = 0x49c - 0x3c0 + slot * 0x5c
            state = struct.unpack_from('<i', player_data, offset+0x40)[0]
            if state in (0, 2):
                continue
            if state != 1:
                raise ValueError('unsupported player shot state')
            address = struct.unpack_from('<I', player_data, offset+0x58)[0]
            if address not in descriptors:
                descriptors[address] = self.block(address, 0x34)
            descriptor = descriptors[address]
            if struct.unpack_from('<I', descriptor, 0x30)[0]:
                raise ValueError('unsupported player shot collision callback')
            position = self.floats(player_data, offset+0x14, 2)
            size = self.floats(descriptor, 0xc, 2)
            if any(v <= 0 or v > 4096 for v in size):
                raise ValueError('invalid player shot hitbox')
            kind = descriptor[0x1d]
            # The native owner skips ordinary shots crossing the top edge.
            if kind != 3 and position[1] - size[1]/2 < 0:
                continue
            shots.append({'slot': slot, 'state': state, 'position': position,
                'velocity_raw': self.floats(player_data, offset+0x20, 2),
                'hitbox_raw': size, 'type': kind,
                'geometry': 'th10-player-shot-aabb-v1'})
        return shots

    def add_acceleration(self, state, manager):
        """Backward velocity difference in pixels/frame^2, not scripted future acceleration."""
        key = (manager, state["stage"], state["stage_frame"])
        previous = self.previous_bullets
        if previous is not None and key == previous[0]:
            for bullet in state["bullets"]:
                old = previous[1].get(bullet["slot"])
                bullet["acceleration"] = old.get("acceleration") if old else None
            return
        dt = key[2]-previous[0][2] if previous is not None and key[:2] == previous[0][:2] else 0
        for bullet in state["bullets"]:
            bullet["acceleration"] = None
            old = previous[1].get(bullet["slot"]) if previous is not None else None
            if (old is not None and 0 < dt <= 2 and old["age_frames"] > 0
                    and bullet["age_frames"]-old["age_frames"] == dt
                    and bullet["type"] == old["type"] and bullet["status"] == old["status"] == 1):
                bullet["acceleration"] = [(v-p)/dt for v, p in zip(bullet["velocity_raw"], old["velocity_raw"])]
        self.previous_bullets = (key, {b["slot"]: b for b in state["bullets"]})
