#!/usr/bin/env bash
set -euo pipefail

KEEP_N=2500
ROOT="$(pwd)"

# 可选：DRY_RUN=1 只打印不删除
DRY_RUN="${DRY_RUN:-0}"

run_rm() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] rm $*"
  else
    rm -rf "$@"
  fi
}

for d in */; do
  traj_dir="$ROOT/$d/trajectory"
  if [[ ! -d "$traj_dir" ]]; then
    echo "⏭️  Skip (no trajectory): $d"
    continue
  fi

  echo "👉 Processing: $traj_dir"
  cd "$traj_dir"

  # 1) 删除 log.pkl 和 .lock 文件
  if [[ -f "log.pkl" ]]; then
    run_rm "log.pkl"
  fi
  # 只删 trajectory 目录下的 *.lock（不递归）
  for f in ./*.lock; do
    [[ -e "$f" ]] || break
    run_rm "$f"
  done

  # 2) 统计 trajectory 下一级的子文件夹数量，并随机删到只剩 KEEP_N
  # 取一级目录（不含 .），用 NUL 分隔避免空格/特殊字符问题
  mapfile -d '' subdirs < <(find . -mindepth 1 -maxdepth 1 -type d -print0)

  total="${#subdirs[@]}"
  echo "   - Found subfolders: $total"

  if (( total <= KEEP_N )); then
    echo "   - <= $KEEP_N, nothing to prune."
    cd "$ROOT"
    continue
  fi

  remove_n=$(( total - KEEP_N ))
  echo "   - Need to remove: $remove_n (keep $KEEP_N)"

  # 随机选 remove_n 个目录删除
  # shuf -z 支持 NUL 分隔（GNU coreutils）
  # 输出仍是 NUL 分隔，while read -d '' 逐个处理
  printf '%s\0' "${subdirs[@]}" \
    | shuf -z -n "$remove_n" \
    | while IFS= read -r -d '' victim; do
        # victim 形如 ./xxx
        run_rm "$victim"
      done

  cd "$ROOT"
done

echo "✅ Done."
