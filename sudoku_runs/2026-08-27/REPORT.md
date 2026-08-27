# 6일차 (2026-08-27) — 스도쿠 완주 런 재개(76k→123k, exact 959) · 재개 무학습 버그 · ARC 전환 준비

> 상세는 루트 `HANDOVER.md`(재구축·재개 절차·함정·오늘 측정). 이 폴더는 로그·코드·체크포인트.

- `R1B8_bilin_r2.log`: 유효 런. 76,167 → 123,039. eval(loops16) 최고 exact 959 @121,086 · 최고 셀 0.8176 @123,039.
- `R1B8_bilin_r2_a.log`, `R1B8_bilin_r2_b_noop.log`: 업스트림 `assign=True` 버그로 **학습이 0이었던** 재개 2회(증거 보존). eval 이 넷째 자리까지 동일한 것이 신호.
- `code/resume27.sh`: 재개 스크립트(EPOCHS=남은 iters×250, CKPT_PATH=파일). `code/urm_patches.diff`: assign 패치 포함 5파일.
- `checkpoints/R1B8_bilin_r2_step123039.pt`: EMA 가중치만(54MB) + config.
- 학습 외: 논문 5편 확보(`papers/`, `PAPERS.md`), ARC-2 데이터 실측·출력크기 규칙·객체 사전분포 적합률, 코어 T=900 벤치 — HANDOVER §4.
