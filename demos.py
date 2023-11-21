import argparse
import utils

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        return
    print(config_path)
    cfg = utils.load_config(config_path)

    # Create env
    env = utils.get_env_from_cfg(cfg, show_gui=True)

    # Create policy
    policy = utils.get_policy_from_cfg(cfg)

    # Run policy
    state = env.reset()
    try:
        while True:
            action = policy.step(state)
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
    finally:
        env.close()

parser = argparse.ArgumentParser()
parser.add_argument('--config-path')
main(parser.parse_args())
