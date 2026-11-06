#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"

printf "%-50s %10s\n" "Render Path" "Count(perm=777)"
printf "%-50s %10s\n" "--------------------------------------------------" "----------"

for d in */; do
  render_dir="$ROOT/$d/render"

  if [[ -d "$render_dir" ]]; then
    cd "$render_dir"

    count=$(find . -maxdepth 1 -type d -perm 777 | grep -v "^\.$" | wc -l)

    printf "%-50s %10d\n" "$d/render" "$count"

    cd "$ROOT"
  fi
done
