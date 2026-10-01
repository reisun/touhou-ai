"""Offline reference only. Reads previously extracted ECL; never imports learner."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/progress-ecl-20260927'
OUTPUT = ROOT / 'docs'
FINAL = {1: 'BossCard2', 2: 'BossCard3', 3: 'BossCard3',
         4: 'BossCard4', 5: 'BossCard4', 6: 'BossCard5'}
BOSSES = {1: '秋穣子', 2: '鍵山雛', 3: '河城にとり', 4: '射命丸文',
          5: '東風谷早苗', 6: '八坂神奈子'}
rows, hashes = [], {}
for stage in range(1, 7):
    path = SOURCE / f'stage{stage:02}.txt'
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    text = path.read_text(encoding='cp932')
    for match in re.finditer(r'void (\w+)\([^\n]*\)\n\{(.*?)\n\}', text, re.S):
        sub, body = match.group(1), match.group(2)
        mask = '*'
        for offset, line in enumerate(body.splitlines(), 1):
            if line.startswith('!'):
                mask = line[1:]
            spell = re.search(r'ins_(342|350|357|358|359)\((\d+), (\d+), (\d+), "(.*)"\);', line)
            if not spell or (mask != '*' and 'N' not in mask):
                continue
            # This alternate routine is reached only by Lunatic from Boss3.
            if sub == 'BossCard3L' or (stage == 1 and sub == 'MBossCard1'):
                continue
            opcode, base, timeout, score = map(int, spell.groups()[:4])
            identifier = base + {342: 0, 350: 0, 357: 1, 358: 0, 359: -1}[opcode]
            role = 'midboss' if sub.startswith('MBoss') else 'boss'
            assert role == 'midboss' or sub.startswith('BossCard'), sub
            final = role == 'midboss' or sub == FINAL[stage]
            callback = ('MBossEscape' if stage == 5 else 'MBossDead') if role == 'midboss' else 'BossDead'
            # Stage 2 midboss inherits its terminal callback from MBoss.
            if final:
                assert f'"{callback}"' in body or (stage == 2 and role == 'midboss'
                    and 'ins_334(0, 0, 2400, "MBossDead")' in text)
            rows.append(dict(stage=stage, difficulty='Normal', difficulty_raw=1,
                role=role, boss=BOSSES[stage], spell_id_raw=identifier,
                # thecl escapes byte 0x5c even when it is a CP932 trail byte.
                name=spell.group(5).replace('\\', ''), subroutine=sub,
                final_for_encounter=final, terminal_callback=callback if final else None,
                opcode=opcode, base_id=base, difficulty_mask=mask,
                source_file=path.relative_to(ROOT).as_posix(),
                source_line=text[:match.start()].count('\n') + offset + 1,
                evidence='pinned_ecl_and_readonly_instruction_check',
                live_completion_verified=False))
rows.sort(key=lambda r: (r['stage'], r['role'] != 'midboss', r['spell_id_raw']))
assert len(rows) == 24, len(rows)
assert len({r['spell_id_raw'] for r in rows}) == 24
assert [r['spell_id_raw'] for r in rows if r['role'] == 'boss' and r['final_for_encounter']] == [7, 23, 39, 54, 74, 94]
assert [r['spell_id_raw'] for r in rows if r['role'] == 'midboss'] == [11, 27, 58]
data = dict(version=1, purpose='reference_only_not_loaded_by_training',
    scope='TH10 pinned data, Normal, stages 1-6; other difficulties and Extra excluded',
    game_data_sha256='1fb1d0ffe34115f563f5feb43755c0feee2315b0ac2b32e2f9e84c81e9433bea',
    extracted_text_sha256=hashes, spells=rows)
(OUTPUT / 'spell-reference-normal.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
lines = ['# 風神録 Normal：スペルID・最終スペル対応表', '',
    '調査日：2026-09-27。現在の学習対象であるNormal・1〜6面に限定。学習への接続なし。', '',
    'IDは現在の `spell.id_raw` と同じ内部番号。攻略サイト等の表示番号とは区別する。',
    '「最終」はその戦闘（中ボス／面ボス）の最後。面クリアを意味するのは面ボス側のみ。', '',
    '| 面 | 区分 | ボス | 内部ID | スペル名 | その戦闘の最終 |',
    '|---|---|---|---:|---|---|']
for r in rows:
    lines.append(f"| {r['stage']} | {'中ボス' if r['role']=='midboss' else '面ボス'} | {r['boss']} | {r['spell_id_raw']} | {r['name']} | {'はい' if r['final_for_encounter'] else '—'} |")
lines += ['', '## 照合方法と確度', '',
    '- 保存済みECL（`artifacts/progress-ecl-20260927/stage01..06.txt`、CP932）から難易度マスクとスペル開始命令を抽出。',
    '- Normalは難易度値1。命令357は基底ID＋難易度、359は基底ID＋難易度−2、342は直接ID。',
    '- 3面の `BossCard3L` はLunatic専用の呼び出しなのでNormal表から除外。',
    '- 最終スペルは `BossDead`／`MBossDead`／`MBossEscape` の終了コールバックを照合。2面中ボスは親の `MBoss` で設定。',
    '- 1面Normalの中ボス、4面中ボスにはスペルなし。6面には中ボスなし。',
    '- 実行中の所有ゲームに対する少量の読み取り専用命令照合で内部IDの格納先を確認。停止・フック追加・入力送信・メモリ書き換えは実施していない。',
    '- `0x410edc..0x410f00`：命令別の難易度補正。`0x410f38`：補正後IDを引数として `0x409280` へ渡す。',
    '- `0x409294`：第2引数のIDをEDIへ。`0x4092e3`：EDIをスペル構造体＋`0x3788`へ格納。既存readerの読取先と一致。',
    '- 全スペルの開始・終了を実プレイで再検証したものではない。静的データと命令の照合結果である。', '',
    '## 公開資料', '',
    '- [thcrap：風神録のスペルID取得設定](https://github.com/thpatch/thcrap-tsa/blob/master/base_tsa/th10.js)',
    '- [thcrap：v1.00aの対応アドレス](https://github.com/thpatch/thcrap-tsa/blob/master/base_tsa/th10.v1.00a.js) — `0x410edc`、`0x410f30`。',
    '- [thprac：風神録の攻撃区間処理](https://github.com/touhouworldcup/thprac/blob/master/thprac/src/thprac/thprac_th10.cpp)',
    '- [Touhou Toolkit](https://github.com/thpatch/thtk) — 既存ECL抽出に使用。', '',
    '## 今後の利用上の注意', '',
    '最終スペルの個別加点を省き、既存の撃破イベント側へ統合するための参照資料。実装・OBS表示は未変更。',
    '最終スペルの時間切れも進行として扱う場合、HP撃破のみを対象とする既存判定の変更が別途必要。',
    'ステージ・難易度・発動中フラグとIDを合わせて照合し、終了後に残るIDだけで達成扱いしない。',
    '他難易度・Extraへこの表を流用しない。', '',
    '再生成・検証：`.venv/Scripts/python.exe scripts/build_spell_reference.py`。保存済みファイルのみを読み、24件の一意性・最終ID・コールバックを検証する。', '']
(OUTPUT / 'spell-reference-normal.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'spells': len(rows), 'boss_final_ids': [7, 23, 39, 54, 74, 94], 'midboss_final_ids': [11, 27, 58], 'training_changed': False}))
