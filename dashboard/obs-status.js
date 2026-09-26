'use strict';
function learningBannerState(frame, stats, fresh) {
    const valid = n => Number.isInteger(n) && n >= 0;
    const saved = stats ? Math.max(0, ...stats.growth.map(e => valid(e.total_updates) ? e.total_updates : 0)) : null;
    const count = fresh && valid(frame?.learning?.updates) ? frame.learning.updates : saved;
    const phase = fresh ? frame?.learning?.phase : null;
    const label = !fresh ? (stats ? '待機中' : '接続待ち') : ({
        playing: 'プレイ中', stage_transition: '会話・面移行中',
        optimizing_at_game_over: '学習更新中', optimizing_recovery: '復旧学習中',
        game_over_ready: '学習完了・待機中', stage_one_ready: '1面待機中',
        recovering: '復旧中', restarting_stage_one: '1面再起動中'
    }[phase] || '観測中');
    return {text: `${label}（${count === null ? '学習回数未取得' : `学習 ${count} 回目の後`}）`,
            playing: phase === 'playing'};
}
