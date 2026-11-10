"""Build an evaluation session around a saved checkpoint (shared by the analysis scripts)."""
from types import SimpleNamespace

import torch

from mamex_robustness import tool
from mamex_robustness.application.environment import Env
from mamex_robustness.application.session import Session
from mamex_robustness.cli import parse_args
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model

DEFAULT_CKPT = 'results/ckpt291/checkpoints/baseline_hashtag_r0.00_g8.0_seed{seed}.pt'


def load_session(seed, ckpt=DEFAULT_CKPT, extra=(), noise_rate=0.4, data_path='processed_data_baby/cold'):
    """Stress-test session for `seed` with the weights in `ckpt` (a path or a pattern with {seed}), in eval mode.

    extra: further command-line arguments, for example ['--warm_only_aux', '1'] or ['--gamma_grid', '1', '2'].
    Returns a namespace with args, env, data, model and session.
    """
    args = tool.apply_mode_defaults(parse_args(['--mode', 'stress_test', '--data_path', data_path, '--seed', str(seed),
                                                '--noise_rate', str(noise_rate), *map(str, extra)]))
    env = Env(args)
    data = Loader4MM(env)
    model = MOME_model(env, data)
    model.load_state_dict(torch.load(ckpt.format(seed=seed), map_location='cpu'))
    model.eval()
    return SimpleNamespace(args=args, env=env, data=data, model=model, session=Session(env, model, data))
