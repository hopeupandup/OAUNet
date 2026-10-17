"""Run modes: formal and verify."""

from utils.paths import RESULT_DIR

RUN_MODES = ('formal', 'verify')
VERIFY_ROOT = RESULT_DIR / 'verify'

# verify-mode overrides; each can also be set from the verify section of the config.
_VERIFY_DEFAULTS = {
    'limit_train': 100,
    'limit_test': 100,
    'epochs': 1,
    'n_tasks': 4,
    'meta_epochs': 1,
    'inner_steps': 1,
    'support_size': 10,
    'query_size': 8,
}


def _log(logger, msg, level='info'):
    if logger is not None and hasattr(logger, level):
        getattr(logger, level)(msg)
    else:
        print(msg)


def apply_run_mode(config, run_mode=None, limit_train=None, limit_test=None,
                   logger=None):
    mode = (run_mode or config.get('run_mode') or 'formal').lower()
    if mode not in RUN_MODES:
        raise ValueError(f"未知 run_mode: {mode}（可选 {RUN_MODES}）")

    config['run_mode'] = mode

    if mode == 'formal':
        config['_limit_train'] = None
        config['_limit_test'] = None
        banner = f"运行模式: formal（正式全量）→ {RESULT_DIR}"
        return {'mode': mode, 'limit_train': None, 'limit_test': None,
                'root': str(RESULT_DIR), 'banner': banner}


    v = dict(_VERIFY_DEFAULTS)
    v.update(config.get('verify', {}) or {})
    lt = int(limit_train if limit_train is not None else v['limit_train'])
    ls = int(limit_test if limit_test is not None else v['limit_test'])

    config.setdefault('paths', {})
    config['paths']['checkpoints_dir'] = str(VERIFY_ROOT / 'checkpoints')
    config['paths']['logs_dir'] = str(VERIFY_ROOT / 'logs')
    config['paths']['results_dir'] = str(VERIFY_ROOT / 'metrics')
    config['paths']['figures_dir'] = str(VERIFY_ROOT / 'figures')
    config['checkpoint_path'] = str(VERIFY_ROOT / 'checkpoints' / 'model_final.pth')

    config.setdefault('training', {})
    config['training']['epochs'] = int(v['epochs'])

    config.setdefault('meta_learning', {})
    config['meta_learning']['n_tasks'] = int(v['n_tasks'])
    config['meta_learning']['meta_epochs'] = int(v['meta_epochs'])
    config['meta_learning']['inner_steps'] = int(v['inner_steps'])
    config['meta_learning']['support_size'] = int(v['support_size'])
    config['meta_learning']['query_size'] = int(v['query_size'])

    config['_limit_train'] = lt
    config['_limit_test'] = ls

    need = (config['meta_learning']['support_size']
            + config['meta_learning']['query_size'])
    group = max(1, lt // max(1, config['meta_learning']['n_tasks']))
    banner = (
        f"运行模式: verify（快速检验）— 训练/测试 {lt}/{ls}，"
        f"epochs={config['training']['epochs']}，"
        f"元学习 {config['meta_learning']['n_tasks']} 任务 × "
        f"{config['meta_learning']['meta_epochs']} 轮，"
        f"support/query={config['meta_learning']['support_size']}/"
        f"{config['meta_learning']['query_size']} → {VERIFY_ROOT}"
    )
    if need > group:
        _log(logger, f"[警告] 元学习 support+query={need} > 每组样本数 {group}"
                     f"（= limit_train // n_tasks），任务会被全部过滤、元训练空转。",
             level='warning')
    return {'mode': mode, 'limit_train': lt, 'limit_test': ls,
            'root': str(VERIFY_ROOT), 'banner': banner}


def add_run_mode_args(parser):
    parser.add_argument('--run_mode', type=str, default=None, choices=list(RUN_MODES),
                        help='运行模式：formal（正式全量，默认）| verify（100 样本快速检验）')
    parser.add_argument('--limit_train', type=int, default=None,
                        help='覆盖 verify 模式的训练样本数（默认 100）')
    parser.add_argument('--limit_test', type=int, default=None,
                        help='覆盖 verify 模式的测试样本数（默认 100）')
