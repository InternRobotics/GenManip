#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"

for d in */; do
  render_dir="$ROOT/$d/render"

  if [[ -d "$render_dir" ]]; then
    echo "👉 Processing: $render_dir"
    cd "$render_dir"

    # 删除 lock 文件
    rm -f *.lock

    # 删除权限不是 777 的一级子目录（不含 .）
    find . -maxdepth 1 -type d ! -perm 777 ! -name "." -print0 | xargs -0 rm -rf

    cd "$ROOT"
  else
    echo "⏭️  Skip (no render): $d"
  fi
done

echo "✅ Done."
