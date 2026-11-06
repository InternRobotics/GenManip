import os
import argparse
import sys
import shutil
import json
import requests
from requests import RequestException

USER_TOKEN = os.getenv("USER_TOKEN", None)
LEADERBOARD_HOST = os.getenv("LEADERBOARD_HOST", "localhost")
LEADERBOARD_PORT = int(os.getenv("LEADERBOARD_PORT", 8000))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def upload_submission(
    host_ip: str,
    port: int,
    submission_file_path: str,
    author_token: str,
    submission_infos: dict,
):
    """
    Upload a submission to the LightBoard server.

    Args:
        host_ip (str): The IP address of the server.
        port (int): The port of the server.
        submission_file_path (str): Path to the .zip file to upload.
        author_token (str): Your authority token (get it from the website dropdown).
        submission_infos (dict): Dictionary containing submission details.
            Must include:
                - 'submission_name': Name of your submission.
                - 'leaderboard_name': Name of the target leaderboard.
            Can include other optional info.
    """
    url = f"http://{host_ip}:{port}/api/upload"

    if not os.path.exists(submission_file_path):
        print(f"Error: File {submission_file_path} does not exist.")
        return None

    submission_name = submission_infos.get("submission_name")
    leaderboard_name = submission_infos.get("leaderboard_name")

    if not submission_name or not leaderboard_name:
        print(
            "Error: submission_infos must contain 'submission_name' and 'leaderboard_name'."
        )
        return None

    files = {
        "file": (
            os.path.basename(submission_file_path),
            open(submission_file_path, "rb"),
            "application/zip",
        )
    }

    data = {
        "submission_name": submission_name,
        "leaderboard_name": leaderboard_name,
        "author_token": author_token,
        "submission_infos": json.dumps(submission_infos),
    }

    try:
        response = requests.post(url, files=files, data=data, timeout=(30, 300))
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "success":
            print(
                f"Successfully uploaded! Submission ID: {result.get('submission_id')}"
            )
            print(
                f"Check status at: http://{host_ip}:{port}/upload_status/{result.get('submission_id')}"
            )
        else:
            print(f"Upload failed: {result.get('error')}")

        return result
    except (RequestException, ValueError, KeyError, TypeError) as e:
        print(f"An error occurred: {e}")
        return {"status": "error", "error": str(e)}


def list_results():
    base_results_dir = os.path.join(PROJECT_ROOT, "saved/eval_results")
    results = []

    if os.path.exists(base_results_dir):
        # Walk through subdirectories to find benchmark_id and run_id
        for benchmark_id in os.listdir(base_results_dir):
            benchmark_dir = os.path.join(base_results_dir, benchmark_id)
            if not os.path.isdir(benchmark_dir):
                continue

            for run_id in os.listdir(benchmark_dir):
                run_dir = os.path.join(benchmark_dir, run_id)
                if os.path.isdir(run_dir):
                    submitted = os.path.exists(os.path.join(run_dir, "submitted.flag"))
                    results.append(
                        {
                            "benchmark_id": benchmark_id,
                            "run_id": run_id,
                            "path": run_dir,
                            "submitted": submitted,
                        }
                    )

    if not results:
        print("No results found.")
    else:
        # ANSI colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"

        print(f"{'Benchmark ID':<30} {'Run ID':<40} {'Submitted':<15} {'Path'}")
        print("-" * 150)
        results = sorted(results, key=lambda x: (x["benchmark_id"], x["run_id"]))
        results.reverse()
        for item in results:
            submitted = item.get("submitted", False)
            status = f"{GREEN}Yes{RESET}" if submitted else f"{RED}No{RESET}"
            print(
                f"{item['benchmark_id']:<30} {item['run_id']:<40} {status:<24} {item['path']}"
            )  # 24 because escape codes take space but are invisible


def submit_results(
    user_token, run_id, submission_name, leaderboard_name, host, port, benchmark_id=None
):
    if benchmark_id:
        results_dir = os.path.join(
            PROJECT_ROOT, "saved/eval_results", benchmark_id, run_id
        )
    else:
        # If benchmark_id is not provided, try to find it
        base_results_dir = os.path.join(PROJECT_ROOT, "saved/eval_results")
        results_dir = None
        if os.path.exists(base_results_dir):
            for b_id in os.listdir(base_results_dir):
                potential_dir = os.path.join(base_results_dir, b_id, run_id)
                if os.path.isdir(potential_dir):
                    results_dir = potential_dir
                    benchmark_id = b_id
                    break

    if results_dir is None or not os.path.exists(results_dir):
        print(f"Error: Results not found for run_id: {run_id}")
        return

    submission_info = {
        "submission_name": submission_name,
        "leaderboard_name": leaderboard_name,
    }

    print(f"Submitting run {run_id} to {leaderboard_name} at {host}:{port}...")

    # Create zip file
    zip_filename = f"results_{run_id}"
    zip_base_path = os.path.join(os.path.dirname(results_dir), zip_filename)

    # shutil.make_archive returns the path to the zip file
    archive_path = shutil.make_archive(zip_base_path, "zip", results_dir)

    try:
        result = upload_submission(
            host,
            port,
            archive_path,
            user_token,
            submission_info,
        )

        if result and result.get("status") == "success":
            # Mark as submitted
            submitted_flag_path = os.path.join(results_dir, "submitted.flag")
            with open(submitted_flag_path, "w") as f:
                json.dump(submission_info, f, indent=4)
        else:
            print("Submission failed.")

    finally:
        # Remove zip file
        if os.path.exists(archive_path):
            os.remove(archive_path)


def main():
    parser = argparse.ArgumentParser(description="Eval Submission Tool")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List available results")

    # Submit command
    submit_parser = subparsers.add_parser(
        "submit", help="Submit a result to the leaderboard"
    )
    submit_parser.add_argument("--run_id", required=True, help="Run ID to submit")
    submit_parser.add_argument(
        "--benchmark_id",
        default=None,
        help="Benchmark ID (optional, will search if not provided)",
    )
    submit_parser.add_argument(
        "-n", "--submission_name", required=True, help="Name of the submission"
    )
    submit_parser.add_argument(
        "-l", "--leaderboard_name", required=True, help="Leaderboard name"
    )
    submit_parser.add_argument(
        "--user_token", default=USER_TOKEN, help="User token for leaderboard"
    )
    submit_parser.add_argument(
        "--host", default=LEADERBOARD_HOST, help="Leaderboard Host"
    )
    submit_parser.add_argument(
        "--port", type=int, default=LEADERBOARD_PORT, help="Leaderboard Port"
    )

    args = parser.parse_args()

    if args.command == "list":
        list_results()
    elif args.command == "submit":
        if args.user_token is None:
            print(
                "Please set USER_TOKEN environment variable or provide it by --user_token."
            )
            sys.exit(1)
        submit_results(
            args.user_token,
            args.run_id,
            args.submission_name,
            args.leaderboard_name,
            args.host,
            args.port,
            benchmark_id=args.benchmark_id,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
