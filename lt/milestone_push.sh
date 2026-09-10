#!/bin/bash
# 학습 런 감시자: 마일스톤 .pt → npz(EMA) 변환·커밋·푸시, 1시간마다 로그 커밋, 학습 종료 시 마지막 푸시.
# 사용: nohup bash lt/milestone_push.sh <TRAIN_PID> <RUN_DIR> <PREFIX> <LOG_FILE> > /dev/null 2>&1 &
#   예: nohup bash lt/milestone_push.sh 12345 runs/v1_2 v12 runs/v1_2/train.log &
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); PID=${1:?}; RUN=${2:?}; PREFIX=${3:?}; LOGF=${4:?}
MDIR=$REPO/$RUN/milestones; NDIR=$REPO/checkpoints; WLOG=$REPO/$RUN/watcher.log
mkdir -p "$NDIR"; say(){ echo "[$(date +%H:%M:%S)] $*" >> "$WLOG"; }
say "watcher 시작 (pid=$PID run=$RUN prefix=$PREFIX)"
commit_push(){ cd "$REPO" || return 1
  git add -A checkpoints/ "$RUN/milestones/"*.txt 2>/dev/null; git add -f "$LOGF" 2>/dev/null
  git diff --cached --quiet && { say "변경 없음"; return 0; }
  STEP=$(grep -o 'step [0-9]*' "$LOGF" 2>/dev/null | tail -1 | grep -o '[0-9]*'); ACC=$(grep "^\[EVAL\]" "$LOGF" 2>/dev/null | tail -1)
  git commit -q -m "$1" -m "step ${STEP:-?} · ${ACC:-eval 없음}" >> "$WLOG" 2>&1
  git push -q origin clean >> "$WLOG" 2>&1 && say "푸시 성공: $1 (step ${STEP:-?})" || say "푸시 실패 — 다음 주기에 재시도"; }
LAST=0
while :; do
  NEW=0; shopt -s nullglob
  for pt in "$MDIR"/step_*.pt; do b=$(basename "$pt" .pt); npz="$NDIR/${PREFIX}_${b}.npz"; [ -f "$npz" ] && continue
    python "$REPO/lt/ckpt_npz.py" pack "$pt" "$npz" >> "$WLOG" 2>&1 && { say "변환 $b"; NEW=1; }; done
  [ $NEW = 1 ] && commit_push "$PREFIX 마일스톤 npz·외삽·로그 (자동)"
  NOW=$(date +%s); [ $((NOW-LAST)) -ge 3600 ] && { commit_push "$PREFIX 학습 로그 (자동)"; LAST=$NOW; }
  kill -0 "$PID" 2>/dev/null || { say "학습 종료 감지 — 마지막 처리"; sleep 20; commit_push "$PREFIX 학습 종료 시점 (자동)"; say "watcher 종료"; exit 0; }
  sleep 120
done
