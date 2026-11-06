# Standalone Tools

This directory contains standalone utility scripts.


## 1. Submission Tool (`submit.py`)

The `submit.py` script provides a command-line interface (CLI) to manage and submit evaluation results to a leaderboard.

### Examples

**List evaluation results:**
```bash
python standalone_tools/submit.py list
```

**Submit a result:**
```bash
export LEADERBOARD_HOST="ip_address_of_leaderboard_server"
export LEADERBOARD_PORT="port_of_leaderboard_server"
export USER_TOKEN="your_secret_token"
python standalone_tools/submit.py submit --run_id "result_run_id_to_submit" --submission_name "My Best Model" --leaderboard_name "EBench"
```

**Submit with explicit token:**
```bash
python submit.py submit --run_id "result_run_id_to_submit" -n "My Best Model" -l "EBench" --user_token "your_secret_token"
```

### Commands

#### `list`
Lists all available evaluation results stored on the server, organized by benchmark, and their submission status.

```bash
python standalone_tools/submit.py list
```

**Output:**
Displays a table with `Benchmark ID`, `Run ID`, `Submitted` status (Green/Red), and the file `Path`.

#### `submit`
Submits a specific evaluation run to the leaderboard. It automatically searches for the `run_id` across benchmark folders if `--benchmark_id` is not provided.

```bash
python standalone_tools/submit.py submit --run_id RUN_ID -n SUBMISSION_NAME -l LEADERBOARD_NAME [OPTIONS]
```

### Arguments

**Global Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | Evaluation server host. | `localhost` |
| `--port` | Evaluation server port. | `8000` |

**Submit Command Arguments:**
| Argument | Description | Required |
|----------|-------------|----------|
| `--run_id` | The ID of the run to submit (from `list` command). | Yes |
| `--benchmark_id` | The ID of the benchmark the run belongs to. If omitted, the tool will search for it. | No |
| `-n`, `--submission_name` | A name for your submission. | Yes |
| `-l`, `--leaderboard_name` | The name of the leaderboard to submit to. | Yes |
| `--user_token` | Your user token for authentication. Can also be set via `USER_TOKEN` env var. | Yes (if env var not set) |
