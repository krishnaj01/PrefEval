#!/usr/bin/env bash
# Run any Phase 2 command inside tmux (survives terminal/SSH disconnects), in the prefeval-p2 conda env.
#
#   scripts/phase2_tmux.sh "python -m phase2.pipeline --stage train"
#   tmux attach -t phase2        # re-attach;  detach with Ctrl-b d
#
# Overrides: CONDA_ENV (default prefeval-p2), SESSION (default phase2).
set -euo pipefail
RUN="${1:?usage: $0 \"<command>\"}"
SESSION="${SESSION:-phase2}"
CONDA_ENV="${CONDA_ENV:-prefeval-p2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
else
  for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "/opt/conda"; do
    [[ -f "$base/etc/profile.d/conda.sh" ]] && { CONDA_SH="$base/etc/profile.d/conda.sh"; break; }
  done
fi
[[ -f "$CONDA_SH" ]] || { echo "Could not find conda.sh; activate $CONDA_ENV yourself and run: $RUN"; exit 1; }
command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed (sudo apt install tmux)."; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (a run may be in progress). Attaching..."
  exec tmux attach -t "$SESSION"
fi
mkdir -p "$REPO/logs/phase2"
LOG="$REPO/logs/phase2/tmux_$(date +%Y%m%d_%H%M%S).log"
CMD="cd '$REPO' && source '$CONDA_SH' && conda activate '$CONDA_ENV' && ( $RUN ) 2>&1 | tee -a '$LOG'"'
 rc=${PIPESTATUS[0]}; echo; echo "== finished (exit $rc). Log: '"$LOG"'"; exec bash'
tmux new-session -d -s "$SESSION" -n run bash -lc "$CMD"
tmux new-window -t "$SESSION" -n gpu "watch -n 5 nvidia-smi"
tmux select-window -t "$SESSION:run"
echo "Started in tmux session '$SESSION' (windows: run, gpu). Log: $LOG"
echo "Attach: tmux attach -t $SESSION   |   Detach: Ctrl-b d"
