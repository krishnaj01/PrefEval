#!/usr/bin/env bash
# Launch (or re-attach to) a Phase 1 run inside tmux so it survives terminal/SSH disconnects.
#
#   scripts/phase1_tmux.sh tierA_pilot        # starts configs/phase1/tierA_pilot.yaml in tmux session "prefeval"
#   tmux attach -t prefeval                   # re-attach later;  detach again with  Ctrl-b  then  d
#
# The run resumes automatically if results already exist (same command = resume).
set -euo pipefail
CONFIG_NAME="${1:?usage: $0 <config name, e.g. tierA_pilot>}"
SESSION="${SESSION:-prefeval}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/phase1/${CONFIG_NAME}.yaml"
[[ -f "$REPO/$CONFIG" ]] || { echo "No such config: $CONFIG"; exit 1; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists (a run may be in progress). Attaching..."
  exec tmux attach -t "$SESSION"
fi

CMD="cd '$REPO' && source ~/anaconda3/etc/profile.d/conda.sh && conda activate prefeval \
 && python -m phase1.runner run --config $CONFIG; \
 echo; echo '== run ended (exit '\$?'). Summarise with: python -m phase1.aggregate =='; exec bash"
tmux new-session -d -s "$SESSION" -n run "bash -lc \"$CMD\""
tmux new-window -t "$SESSION" -n gpu "watch -n 5 nvidia-smi"
tmux select-window -t "$SESSION:run"
echo "Started '$CONFIG' in tmux session '$SESSION' (windows: run, gpu)."
echo "Attach: tmux attach -t $SESSION   |   Detach: Ctrl-b d   |   Switch window: Ctrl-b n"
