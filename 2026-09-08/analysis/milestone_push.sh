#!/bin/bash
# v1.6 학습 마일스톤 자동 커밋·푸시.
#   · 새 milestones/step_N.pt 가 생기면 fp16 npz(~13MB)로 변환해 커밋 (.pt 는 143MB, GitHub 100MB 제한 초과)
#   · 외삽 txt·학습 로그도 함께 커밋
#   · 마일스톤이 없어도 1시간마다 로그만 커밋 (서버가 죽어도 곡선은 남게)
#   · 학습 프로세스가 사라지면 마지막으로 한 번 더 푸시하고 종료
# 사용: nohup bash 2026-09-08/analysis/milestone_push.sh <TRAIN_PID> > /dev/null 2>&1 &
set -u
REPO=/workspace/LinearTuring
MDIR=$REPO/2026-09-08/run_v16/milestones
NDIR=$REPO/2026-09-08/checkpoints
LOG=$REPO/2026-09-08/results/milestone_push.log
TRAIN_PID=${1:-0}
mkdir -p "$NDIR" "$(dirname "$LOG")"
say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
say "watcher 시작 (train pid=$TRAIN_PID)"

commit_push() {   # $1 = 커밋 메시지 제목
  cd "$REPO" || return 1
  cp -f "$REPO/2026-09-08/train_v16.log" "$REPO/2026-09-08/train_v16.log" 2>/dev/null
  git add -A 2026-09-08/ >> "$LOG" 2>&1
  if git diff --cached --quiet; then say "변경 없음 — 건너뜀"; return 0; fi
  local STEP ACC
  STEP=$(cat 2026-09-08/train_v16*.log 2>/dev/null | grep -o 'step [0-9]*' | tail -1 | grep -o '[0-9]*')
  ACC=$(cat 2026-09-08/train_v16*.log 2>/dev/null | grep "^\[EVAL\]" | tail -1)
  git commit -q -m "$1" -m "step ${STEP:-?} · ${ACC:-eval 없음}" \
    -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
    -m "Claude-Session: https://claude.ai/code/session_013S3A18eaYkivG5KAc1LbNW" >> "$LOG" 2>&1
  if git push origin clean >> "$LOG" 2>&1; then say "푸시 성공: $1 (step ${STEP:-?})"
  else say "푸시 실패 — 커밋은 로컬에 남음. 다음 주기에 재시도"; fi
}

LAST_HEARTBEAT=0
while true; do
  NEW=0
  shopt -s nullglob
  for pt in "$MDIR"/step_*.pt; do
    base=$(basename "$pt" .pt)                      # step_100000
    npz="$NDIR/v16_${base}.npz"
    [ -f "$npz" ] && continue
    # 쓰기 중일 수 있으니 최근 120초 내 수정된 파일은 건너뛴다
    [ $(( $(date +%s) - $(stat -c %Y "$pt") )) -lt 120 ] && { say "$base 아직 쓰는 중 — 대기"; continue; }
    say "$base → npz 변환 시작"
    if python "$REPO/checkpoints/ckpt_npz.py" pack "$pt" "$npz" \
         --note "v1.6 (post·상수게이지·agree제거·잔차흔적). train_v16.py" >> "$LOG" 2>&1; then
      say "$base 변환 완료 ($(( $(stat -c %s "$npz") / 1000000 ))MB)"; NEW=1
    else
      say "$base 변환 실패"; rm -f "$npz"
    fi
  done
  shopt -u nullglob

  NOW=$(date +%s)
  if [ "$NEW" = "1" ]; then
    commit_push "v1.6 마일스톤 체크포인트(npz)·외삽·로그"
    LAST_HEARTBEAT=$NOW
  elif [ $(( NOW - LAST_HEARTBEAT )) -ge 3600 ]; then
    commit_push "v1.6 학습 로그 (자동)"
    LAST_HEARTBEAT=$NOW
  fi

  if [ "$TRAIN_PID" != "0" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    say "학습 프로세스 종료 감지 — 마지막 처리"
    sleep 180                                        # 마지막 저장이 끝나기를 기다림
    shopt -s nullglob
    for pt in "$MDIR"/step_*.pt; do
      base=$(basename "$pt" .pt); npz="$NDIR/v16_${base}.npz"
      [ -f "$npz" ] || python "$REPO/checkpoints/ckpt_npz.py" pack "$pt" "$npz" --note "v1.6 최종" >> "$LOG" 2>&1
    done
    shopt -u nullglob
    commit_push "v1.6 학습 종료 — 최종 체크포인트·로그"
    say "watcher 종료"; exit 0
  fi
  sleep 300
done
