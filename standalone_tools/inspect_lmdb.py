"""
Inspect GenManip LMDB trajectory/render data.

Examples:
python3 standalone_tools/inspect_lmdb.py --data_path path/to/episode --list-groups
python3 standalone_tools/inspect_lmdb.py --data_path path/to/episode --group scalar_data
python3 standalone_tools/inspect_lmdb.py --data_path path/to/episode --key gripper_close
python3 standalone_tools/inspect_lmdb.py --data_path path/to/episode --key observation/robot/qpos
"""

import argparse
import pickle
from pprint import pformat
from pathlib import Path
from typing import Any

import lmdb
import numpy as np


def load_meta_info(data_path: Path) -> dict[str, Any]:
    with open(data_path / "meta_info.pkl", "rb") as f:
        return pickle.load(f)


def open_lmdb(data_path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(data_path / "lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def to_bytes_key(key: str | bytes) -> bytes:
    if isinstance(key, bytes):
        return key
    return key.encode("utf-8")


def summarize_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return (
            f"ndarray shape={value.shape} dtype={value.dtype} values={value.tolist()}"
        )
    if isinstance(value, list):
        return f"list len={len(value)} values={pformat(value, compact=False)}"
    if isinstance(value, tuple):
        return f"tuple len={len(value)} values={pformat(value, compact=False)}"
    if isinstance(value, dict):
        return f"dict values={pformat(value, compact=False)}"
    return repr(value)


def print_group_keys(meta_info: dict[str, Any], group: str) -> None:
    keys = meta_info["keys"][group]
    print(f"[{group}] total={len(keys)}")
    for idx, key in enumerate(keys):
        printable = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        print(f"{idx:04d}: {printable}")


def read_pickled_value(env: lmdb.Environment, lmdb_key: bytes) -> Any:
    with env.begin(write=False) as txn:
        raw = txn.get(lmdb_key)
    if raw is None:
        raise KeyError(f"LMDB key not found: {lmdb_key!r}")
    return pickle.loads(raw)


def resolve_key(meta_info: dict[str, Any], key: str | bytes) -> tuple[str, bytes]:
    target_key = to_bytes_key(key)
    for group, keys in meta_info["keys"].items():
        if not isinstance(keys, list):
            continue
        for stored_key in keys:
            stored_key_bytes = to_bytes_key(stored_key)
            if stored_key_bytes == target_key:
                return group, stored_key_bytes
    raise KeyError(f"Key not found in meta_info['keys']: {key!r}")


def print_sequence_samples(value: Any, limit: int | None) -> None:
    if isinstance(value, (list, tuple)):
        print(f"type={type(value).__name__} len={len(value)}")
        sequence = value if limit is None else value[:limit]
        for idx, item in enumerate(sequence):
            print(f"[{idx}] {summarize_value(item)}")
        return
    if isinstance(value, np.ndarray) and value.ndim > 0:
        print(f"type=ndarray shape={value.shape} dtype={value.dtype}")
        sequence = value if limit is None else value[:limit]
        for idx, item in enumerate(sequence):
            print(f"[{idx}] {summarize_value(item)}")
        return
    print(f"type={type(value).__name__}")
    print(summarize_value(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List available meta_info key groups.",
    )
    parser.add_argument(
        "--group",
        type=str,
        help="Print all keys under meta_info['keys'][group].",
    )
    parser.add_argument(
        "--key",
        type=str,
        help="Read and inspect a single logical key, e.g. gripper_close.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many sequence items to print. Default prints all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    meta_info = load_meta_info(data_path)

    if args.list_groups:
        print("Available groups:")
        for group, keys in meta_info["keys"].items():
            if isinstance(keys, list):
                print(f"- {group}: {len(keys)} keys")
            else:
                print(f"- {group}: {type(keys).__name__}")

    if args.group:
        if args.group not in meta_info["keys"]:
            raise KeyError(f"Unknown group: {args.group}")
        print_group_keys(meta_info, args.group)

    if args.key:
        env = open_lmdb(data_path)
        group, lmdb_key = resolve_key(meta_info, args.key)
        value = read_pickled_value(env, lmdb_key)
        printable = lmdb_key.decode("utf-8", errors="replace")
        print(f"group={group}")
        print(f"key={printable}")
        print_sequence_samples(value, args.limit)

    if not args.list_groups and not args.group and not args.key:
        print("Nothing to do. Use --list-groups, --group, or --key.")


if __name__ == "__main__":
    main()
