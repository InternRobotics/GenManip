import time
import argparse

from genmanip_client import EvalClient, fake_action


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_ids",
        type=lambda s: s.split(","),
        default=["0"],
        help="List of worker IDs, i.e. --worker_ids 0,1,2",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-a", "--arm_type", type=str, default="franka")
    parser.add_argument("-g", "--gripper_type", type=str, default="panda_hand")
    parser.add_argument("-c", "--control_type", type=str, default="joint_position")
    parser.add_argument(
        "--frame_save_interval",
        type=int,
        default=0,
        help="Save one image every N steps (0 disables, default: 0)",
    )
    parser.add_argument(
        "--plot_on_episode_end",
        action="store_true",
        help="Run 'gmp plot' asynchronously after each finished episode",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1,
        help="Number of actions sent per request (default: 1). >1 enables chunked stepping.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    host = args.host
    port = args.port
    worker_ids = args.worker_ids
    base_url = f"http://{host}:{port}"

    # Create workers on server here, make sure they are created before stepping
    client = EvalClient(
        base_url,
        worker_ids=worker_ids,
        frame_save_interval=args.frame_save_interval,
        plot_on_episode_end=args.plot_on_episode_end,
    )

    # wrap the eval loop in a try-finally to ensure cleanup
    try:
        chunk_size = int(args.chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        obs = client.reset()

        while True:

            action = {
                i: fake_action(
                    args.arm_type,
                    args.gripper_type,
                    args.control_type,
                    chunk_size=chunk_size,
                )
                for i in worker_ids
            }

            obs, done = client.step(action)

            if done:
                # finished all evaluations
                break
            if obs is None:
                break
            # Check if obs data is valid before accessing
            worker_obs = obs.get(worker_ids[0], {}).get("obs")
            if worker_obs is None:
                # No valid observation (e.g., server finished all tasks)
                break
            if worker_obs.get("reset"):
                # model.reset()
                pass
    finally:
        client.close()
