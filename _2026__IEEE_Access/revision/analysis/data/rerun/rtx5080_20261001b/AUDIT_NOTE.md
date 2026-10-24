# 감사 경위 — `rtx5080_20261001b`

실험 코드는 `e1ce911`(`RUN_INFO.md`)입니다. `RESULTS.md`는 감사 스크립트를 고친 `d171057`에서 다시 생성했습니다. 감사는 데이터를 읽기만 하므로 실험은 다시 돌리지 않았습니다.

## 무슨 일이 있었나

1. **1차 감사 (09-29 09:10): NOT CLEAN.** 원인은 두 가지였습니다.
   - PDDLStream 15개 배치가 1초 만에 `incomplete`. 러너가 import하는 PDDLStream 파일(`motion_planners`, 업스트림 예제, 로봇 모델)이 git에 없어 클론에 빠져 있었습니다. 원래 머신(`rci-pc`)에서 로컬에 없는 파일만 옮겼고(기존 파일은 덮어쓰지 않음), 옮긴 뒤 내용 비교에서 차이 0이었습니다.
   - cuTAMP 7시행이 GPU 공유로 플래그됨: Move rep0 seed 0, Stir rep0 seed 6·21, Transfer 정답 rep0 seed 26, 인식 rep0 seed 24, rep1 seed 8, rep2 seed 20. RERUN.md §7대로 지우고(`*.csv.flagged-1790640699`에 원본 보존) 같은 태그로 다시 돌렸습니다(09:12–09:25). 이어서 PDDLStream 전체를 돌렸습니다(09:25–10:57).
2. **2차 감사 (10:57): NOT CLEAN.** 같은 7개 외부 샘플이 **한 시드씩 밀려** 다시 플래그됐습니다(0→1, 6→7, 21→22, 26→27, 24→25, 8→9, 20→21).
3. **원인은 감사 스크립트였습니다.** cuTAMP 시행 구간을 "직전 행 기록 시각부터"로 잡아서 앞 시행의 정리 시간이 다음 시행에 들어갔고, `--drop-flagged` 뒤에는 지운 시행의 시간대가 다음 행으로 넘어갔습니다.
   - 외부 샘플 6건 중 5건은 이 캠페인의 시뮬레이터가 종료되던 순간입니다. pid가 그 시행의 `sim pgid`와 같고, 종료 중이라 명령줄을 읽지 못해 외부로 잡혔습니다.
   - 1건(06:33:06)은 실제 외부 프로세스였습니다. RustDesk 원격 접속의 코덱 점검이 GPU 264 MiB를 썼습니다. 다만 seed 23 기록(06:32:01)과 seed 24 시작(06:33:22) 사이, 즉 시행과 시행 사이였습니다.
4. **조치 (저자 결정).** 구간을 그 시행 자신의 시작 시각(`sim_seed<N>_<시각>.log`)부터 잡도록 감사를 고쳤습니다(`d171057`). 고친 기준으로는 원래 행과 재실행 행 모두 어느 시행도 GPU를 공유하지 않았습니다. 그래서 오염되지 않았는데 지워진 **원래 행을 복원**했고, 재실행 행은 `*.csv.rerun-superseded-1790649976`에 보존했습니다. 이 결정은 두 쪽 결과를 비교하기 전에 내렸습니다.

## 원래 행 대 재실행 행 (결과로 쓰는 것은 원래 행)

| 배치 | seed | 원래: 작업 성공 / 계획 [s] | 재실행: 작업 성공 / 계획 [s] |
|---|---|---|---|
| move_ground_truth_rep0 | 0 | True / 8.85 | True / 8.81 |
| stir_ground_truth_rep0 | 6 | True / 25.18 | True / 24.82 |
| stir_ground_truth_rep0 | 21 | True / 11.80 | True / 11.82 |
| transfer_ground_truth_rep0 | 26 | True / 14.55 | True / 43.50 |
| transfer_perception_rep0 | 24 | True / 14.22 | True / 14.86 |
| transfer_perception_rep1 | 8 | True / 43.86 | True / 14.33 |
| transfer_perception_rep2 | 20 | False / 43.92 | False / 43.75 |
