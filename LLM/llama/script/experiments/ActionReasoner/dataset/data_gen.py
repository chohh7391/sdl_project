"""
ActionReasoner 학습 데이터 생성 스크립트
규칙 문서에 따라 전체 유효 조합을 생성한 뒤 균등 분포로 샘플링하여 JSON 저장
"""

import json
import random
import itertools
from collections import defaultdict
from copy import deepcopy

random.seed(42)

# =============================================================================
# 1. 환경 설정
# =============================================================================

ALL_GRIDS = [f"G{i}" for i in range(1, 13)]  # G1~G12
PLACEABLE_GRIDS = [g for g in ALL_GRIDS if g != "G2"]  # 초기 배치 가능 그리드
WORKSPACE_EDGE = ["G7", "G8", "G12"]

OBJECT_TYPES = {
    "vessel": ["beaker", "flask"],
    "object": ["box", "bottle"],
}

INSTANCES = {}
for category, types in OBJECT_TYPES.items():
    for t in types:
        INSTANCES[t] = [t, f"{t}_A", f"{t}_B", f"{t}_C"]

ALL_VESSELS = []
for t in OBJECT_TYPES["vessel"]:
    ALL_VESSELS.extend(INSTANCES[t])

ALL_OBJECTS = []
for t in OBJECT_TYPES["object"]:
    ALL_OBJECTS.extend(INSTANCES[t])

ALL_ITEMS = ALL_VESSELS + ALL_OBJECTS

VESSEL_TYPES = set(OBJECT_TYPES["vessel"])
OBJECT_TYPE_SET = set(OBJECT_TYPES["object"])


def get_base_type(obj_name):
    """beaker_A -> beaker, box -> box"""
    for t in ["beaker", "flask", "box", "bottle"]:
        if obj_name == t or obj_name.startswith(t + "_"):
            return t
    return None


def is_vessel(obj_name):
    return get_base_type(obj_name) in VESSEL_TYPES


def is_object(obj_name):
    return get_base_type(obj_name) in OBJECT_TYPE_SET


def get_tool_for_obj(obj_name):
    """객체 유형에 따른 도구 반환"""
    if is_vessel(obj_name):
        return "dh3"
    else:
        return "vgc10"


# =============================================================================
# 2. 그리드 인접 관계 정의 (사진 기반)
# =============================================================================
# 레이아웃:
#   G1  [robot] G2
#   G3  G4  G5  G6  G7
#   G8  G9  G10 G11 G12

ADJACENCY = {
    "G1":  ["G3", "G4"],
    "G2":  ["G6", "G7"],
    "G3":  ["G1", "G4", "G8", "G9"],
    "G4":  ["G1", "G3", "G5", "G8", "G9"],
    "G5":  ["G4", "G6", "G9", "G10"],
    "G6":  ["G2", "G5", "G7", "G10", "G11"],
    "G7":  ["G2", "G6", "G11", "G12"],
    "G8":  ["G3", "G4", "G9"],
    "G9":  ["G3", "G4", "G5", "G8", "G10"],
    "G10": ["G5", "G6", "G9", "G11"],
    "G11": ["G6", "G7", "G10", "G12"],
    "G12": ["G7", "G11"],
}

# Transfer용: 대상의 한쪽 옆 2개 그리드 쌍
# 각 그리드에서 Transfer 시 동시에 비어야 하는 2그리드 쌍 목록
TRANSFER_CLEAR_PAIRS = {
    "G1":  [("G3", "G4")],
    "G3":  [("G1", "G4"), ("G8", "G9"), ("G4", "G9")],
    "G4":  [("G1", "G3"), ("G3", "G5"), ("G8", "G9"), ("G5", "G9")],
    "G5":  [("G4", "G6"), ("G9", "G10"), ("G4", "G9"), ("G6", "G10")],
    "G6":  [("G5", "G7"), ("G10", "G11"), ("G5", "G10"), ("G7", "G11")],
    "G7":  [("G6", "G12"), ("G11", "G12"), ("G6", "G11")],
    "G8":  [("G3", "G9"), ("G3", "G4")],
    "G9":  [("G8", "G10"), ("G4", "G5"), ("G3", "G10"), ("G4", "G10")],
    "G10": [("G9", "G11"), ("G5", "G6"), ("G5", "G11"), ("G9", "G6")],
    "G11": [("G10", "G12"), ("G6", "G7"), ("G6", "G12"), ("G10", "G7")],
    "G12": [("G7", "G11")],
    "G2":  [("G6", "G7")],
}


# =============================================================================
# 3. XDL 연산자 정의
# =============================================================================

OPERATORS = ["Add", "Stir", "HeatChill", "Transfer", "CleanVessel", "Move"]

# 수치 파라미터 범위
VOLUMES = [f"{v} mL" for v in range(1, 21)]
TEMPS = [f"{t} C" for t in range(0, 21)]
STIR_TIMES = [f"{s} s" for s in range(1, 31)]
REAGENTS = ["water", "ethanol"]


def get_main_tool(operator, target_obj=None):
    if operator == "Transfer":
        return "ag95"
    elif operator == "Move" and target_obj and is_object(target_obj):
        return "vgc10"
    else:
        return "dh3"


def generate_xdl_string(operator, targets, params=None):
    """연산자와 대상으로 XDL 문자열 생성"""
    if operator == "Add":
        reagent = random.choice(REAGENTS)
        volume = random.choice(VOLUMES)
        return f'<Add vessel="{targets[0]}" reagent="{reagent}" volume="{volume}" />'
    elif operator == "Stir":
        time = random.choice(STIR_TIMES)
        return f'<Stir vessel="{targets[0]}" time="{time}" />'
    elif operator == "HeatChill":
        temp = random.choice(TEMPS)
        return f'<HeatChill vessel="{targets[0]}" temp="{temp}" active="true" />'
    elif operator == "Transfer":
        volume = random.choice(VOLUMES)
        return f'<Transfer from_vessel="{targets[0]}" to_vessel="{targets[1]}" volume="{volume}" />'
    elif operator == "CleanVessel":
        return f'<CleanVessel vessel="{targets[0]}" />'
    elif operator == "Move":
        place = random.choice(["plate_A", "plate_B"])
        return f'<Move object="{targets[0]}" place="{place}" />'
    return ""


def get_operator_target_count(operator):
    if operator == "Transfer":
        return 2
    elif operator == "Move":
        return 1
    else:
        return 1


def get_valid_targets(operator):
    """연산자에 유효한 대상 객체 목록 반환"""
    if operator in ["Add", "Stir", "HeatChill", "CleanVessel"]:
        return ALL_VESSELS
    elif operator == "Transfer":
        return ALL_VESSELS  # from/to 모두 vessel
    elif operator == "Move":
        return ALL_ITEMS  # vessel + object 모두
    return []


# =============================================================================
# 4. Blocking 판정
# =============================================================================

def get_blocking_obstacles(operator, target_grid, occupied_grids, target_grids_set):
    """
    주어진 연산자와 대상 그리드에서, 점유 중인 그리드 중 blocking하는 것 반환
    target_grids_set: Current XDL 대상 객체가 점유한 그리드 집합 (본인 제외)
    """
    blockers = []

    if operator == "Transfer":
        # 인접 2그리드 쌍 중 하나라도 완전히 비어있으면 OK
        pairs = TRANSFER_CLEAR_PAIRS.get(target_grid, [])
        all_blocked = True
        for pair in pairs:
            g1, g2 = pair
            if g1 not in occupied_grids and g2 not in occupied_grids:
                all_blocked = False
                break
        if all_blocked and pairs:
            # 모든 쌍이 막혀있음 -> 각 쌍에서 blocking하는 그리드 수집
            for pair in pairs:
                for g in pair:
                    if g in occupied_grids and g not in target_grids_set:
                        if g not in blockers:
                            blockers.append(g)
    else:
        # 인접 1그리드 중 하나라도 비어있으면 OK
        adj = ADJACENCY.get(target_grid, [])
        all_adj_occupied = all(
            a in occupied_grids for a in adj if a not in target_grids_set
        )
        if all_adj_occupied and adj:
            for a in adj:
                if a in occupied_grids and a not in target_grids_set:
                    if a not in blockers:
                        blockers.append(a)

    return blockers


def is_blocking(operator, target_grid, obstacle_grid, occupied_grids, target_grids_set):
    """특정 장애물이 blocking하는지 확인"""
    if operator == "Transfer":
        pairs = TRANSFER_CLEAR_PAIRS.get(target_grid, [])
        for pair in pairs:
            g1, g2 = pair
            occ1 = g1 in occupied_grids and g1 not in target_grids_set
            occ2 = g2 in occupied_grids and g2 not in target_grids_set
            if not occ1 and not occ2:
                return False  # 이 쌍은 비어있으므로 접근 가능
            if (g1 == obstacle_grid and not occ2) or (g2 == obstacle_grid and not occ1):
                # 이 장애물을 치우면 이 쌍이 열림
                pass
        # 장애물이 인접 쌍에 포함되어야 blocking
        for pair in pairs:
            if obstacle_grid in pair:
                return True
        return False
    else:
        adj = ADJACENCY.get(target_grid, [])
        return obstacle_grid in adj


# =============================================================================
# 5. Candidate Grids 생성
# =============================================================================

def generate_candidate_grids(occupied_grids, obstacle_grid, target_grids,
                              next_operator):
    """후보 그리드 생성 및 카테고리 분류"""
    excluded = set(occupied_grids) | set(target_grids)
    if obstacle_grid:
        excluded.discard(obstacle_grid)  # 장애물 자체 그리드는 이동하므로 비게 됨

    available = [g for g in ALL_GRIDS if g not in excluded]

    # G2 처리: 장애물을 치운 후 G2가 비게 되는 경우도 고려
    if "G2" in excluded and obstacle_grid != "G2":
        pass  # G2 점유 유지
    elif "G2" not in excluded or obstacle_grid == "G2":
        if "G2" not in available:
            available.append("G2")

    categories = {}

    # stirrer/heater zone
    if "G2" in available:
        if next_operator == "Stir":
            categories["stirrer zone"] = ["G2"]
            available.remove("G2")
        elif next_operator == "HeatChill":
            categories["heater zone"] = ["G2"]
            available.remove("G2")
        else:
            pass  # G2는 open area로 분류

    # plate zone
    if "G1" in available:
        categories["plate zone"] = ["G1"]
        available.remove("G1")

    # workspace edge
    edges = [g for g in WORKSPACE_EDGE if g in available]
    if edges:
        categories["workspace edge"] = edges
        available = [g for g in available if g not in edges]

    # open area
    if available:
        categories["open area"] = available

    return categories


def format_candidate_grids(categories):
    """카테고리 딕셔너리를 프롬프트 문자열로 변환"""
    lines = []
    for cat, grids in categories.items():
        lines.append(f"{cat}: [{', '.join(grids)}]")
    return "\n".join(lines)


def count_candidates(categories):
    return sum(len(v) for v in categories.values())


# =============================================================================
# 6. 목적지 결정
# =============================================================================

def determine_target_grid(obstacle_name, next_operator, next_targets,
                           categories):
    """재배치 목적지 결정"""
    is_next_target = next_targets and obstacle_name in next_targets

    if is_next_target:
        # 문맥 인식 재배치
        if next_operator == "Stir" and "stirrer zone" in categories:
            return random.choice(categories["stirrer zone"])
        elif next_operator == "HeatChill" and "heater zone" in categories:
            return random.choice(categories["heater zone"])
        elif next_operator == "Transfer":
            if "open area" in categories:
                return random.choice(categories["open area"])
            elif "workspace edge" in categories:
                return random.choice(categories["workspace edge"])
        else:
            # Add, CleanVessel, Move -> open area
            if "open area" in categories:
                return random.choice(categories["open area"])
            elif "workspace edge" in categories:
                return random.choice(categories["workspace edge"])
    else:
        # 무관한 장애물 -> workspace edge 우선
        if "workspace edge" in categories:
            return random.choice(categories["workspace edge"])
        elif "open area" in categories:
            return random.choice(categories["open area"])
        elif "plate zone" in categories:
            return random.choice(categories["plate zone"])

    # fallback
    all_grids = []
    for grids in categories.values():
        all_grids.extend(grids)
    return random.choice(all_grids) if all_grids else None


# =============================================================================
# 7. 데이터 생성 메인 로직
# =============================================================================

def generate_data():
    all_data = []
    data_by_category = defaultdict(list)

    for cur_op in OPERATORS:
        for next_op_raw in OPERATORS + [None]:
            next_op = next_op_raw

            # Stir-HeatChill 동시 등장 제외
            if (cur_op == "Stir" and next_op == "HeatChill") or \
               (cur_op == "HeatChill" and next_op == "Stir"):
                continue

            # Current 대상 객체 생성
            cur_valid = get_valid_targets(cur_op)
            cur_target_count = get_operator_target_count(cur_op)

            # 대상 객체 조합 샘플링 (전수는 너무 많으므로 유형별 대표 샘플)
            cur_target_samples = []
            if cur_target_count == 1:
                for obj in cur_valid:
                    cur_target_samples.append([obj])
            else:  # Transfer: 2개, 순서 있음
                for o1 in cur_valid:
                    for o2 in cur_valid:
                        if o1 != o2:
                            cur_target_samples.append([o1, o2])

            # 샘플 수 제한
            if len(cur_target_samples) > 8:
                cur_target_samples = random.sample(cur_target_samples, 8)

            # Next 대상 객체
            if next_op is None:
                next_target_samples = [[]]
            else:
                next_valid = get_valid_targets(next_op)
                next_target_count = get_operator_target_count(next_op)
                if next_target_count == 1:
                    next_target_samples = [[obj] for obj in next_valid]
                else:
                    next_target_samples = []
                    for o1 in next_valid:
                        for o2 in next_valid:
                            if o1 != o2:
                                next_target_samples.append([o1, o2])
                if len(next_target_samples) > 6:
                    next_target_samples = random.sample(next_target_samples, 6)

            for cur_targets in cur_target_samples:
                for next_targets in next_target_samples:
                    # 모든 대상 객체 수집 (중복 제거)
                    all_target_objs = list(set(cur_targets + next_targets))

                    # 물체 수 체크: 대상 + 장애물 <= 4
                    max_obstacles = 4 - len(all_target_objs)

                    # --- 케이스 A: 재배치 불필요 (장애물 없음) ---
                    try:
                        entry = generate_no_obstacle_entry(
                            cur_op, cur_targets, next_op, next_targets,
                            all_target_objs
                        )
                        if entry:
                            all_data.append(entry)
                            data_by_category["no_rearrange_no_obstacle"].append(entry)
                    except Exception:
                        pass

                    if max_obstacles <= 0:
                        continue

                    # --- 케이스 B-1: 문맥 인식 재배치 (장애물 = Next 대상 객체) ---
                    if next_targets:
                        for obs_obj in next_targets:
                            if obs_obj not in cur_targets:
                                try:
                                    entry = generate_blocking_entry(
                                        cur_op, cur_targets, next_op, next_targets,
                                        all_target_objs, obs_obj
                                    )
                                    if entry:
                                        all_data.append(entry)
                                        data_by_category["rearrange_context"].append(entry)
                                except Exception:
                                    pass

                    # 장애물 후보 객체 (대상과 다른 객체)
                    obstacle_candidates = [o for o in ALL_ITEMS
                                           if o not in all_target_objs]
                    if len(obstacle_candidates) > 6:
                        obstacle_candidates = random.sample(obstacle_candidates, 6)

                    for obs_obj in obstacle_candidates:
                        # --- 케이스 B-2: 무관한 장애물 + blocking ---
                        try:
                            entry = generate_blocking_entry(
                                cur_op, cur_targets, next_op, next_targets,
                                all_target_objs, obs_obj
                            )
                            if entry:
                                all_data.append(entry)
                                data_by_category["rearrange_irrelevant"].append(entry)
                        except Exception:
                            pass

                        # --- 케이스 C: 장애물 존재 + not blocking ---
                        try:
                            entry = generate_not_blocking_entry(
                                cur_op, cur_targets, next_op, next_targets,
                                all_target_objs, obs_obj
                            )
                            if entry:
                                all_data.append(entry)
                                data_by_category["no_rearrange_not_blocking"].append(entry)
                        except Exception:
                            pass

    return all_data, data_by_category


def generate_no_obstacle_entry(cur_op, cur_targets, next_op, next_targets,
                                all_target_objs):
    """장애물이 없는 경우 데이터 생성"""
    # 대상 객체를 그리드에 배치
    grids = random.sample(PLACEABLE_GRIDS, len(all_target_objs))
    obj_grid_map = dict(zip(all_target_objs, grids))
    occupied = set(grids)

    # main_tool
    main_tool = get_main_tool(cur_op, cur_targets[0] if cur_targets else None)

    # XDL 문자열
    cur_xdl = generate_xdl_string(cur_op, cur_targets)
    next_xdl = generate_xdl_string(next_op, next_targets) if next_op else "None"

    # candidate grids
    target_grids = list(obj_grid_map.values())
    categories = generate_candidate_grids(
        occupied, None, target_grids, next_op
    )
    if count_candidates(categories) < 3:
        return None

    return {
        "instruction": {
            "current_xdl": cur_xdl,
            "next_xdl": next_xdl,
            "obstacle_info": "None",
            "candidate_grids": format_candidate_grids(categories),
        },
        "output": f"{main_tool},False,None,None"
    }


def generate_blocking_entry(cur_op, cur_targets, next_op, next_targets,
                             all_target_objs, obs_obj):
    """blocking 장애물이 있는 경우 데이터 생성"""
    # 대상 객체 배치
    available_grids = list(PLACEABLE_GRIDS)
    random.shuffle(available_grids)

    obj_grid_map = {}
    for obj in all_target_objs:
        if not available_grids:
            return None
        g = available_grids.pop(0)
        obj_grid_map[obj] = g

    # 장애물을 Current 대상의 인접 그리드에 배치 (blocking 보장)
    primary_target_grid = obj_grid_map[cur_targets[0]]
    adj_grids = ADJACENCY.get(primary_target_grid, [])
    target_grids_set = set(obj_grid_map.values())

    blocking_candidates = [g for g in adj_grids
                           if g not in target_grids_set and g != "G2"]
    if not blocking_candidates:
        return None

    obs_grid = random.choice(blocking_candidates)
    obj_grid_map[obs_obj] = obs_grid
    occupied = set(obj_grid_map.values())

    # blocking 확인
    if not is_blocking(cur_op, primary_target_grid, obs_grid,
                       occupied, target_grids_set):
        # 단일 인접이 blocking 안 될 수 있음 (다른 인접이 비어있으면)
        # Transfer가 아닌 경우, 다른 인접 그리드가 비어있으면 not blocking
        if cur_op != "Transfer":
            other_adj = [a for a in adj_grids
                         if a not in occupied and a not in target_grids_set]
            if other_adj:
                return None  # not blocking

    # main_tool, aux_tool
    main_tool = get_main_tool(cur_op, cur_targets[0] if cur_targets else None)
    aux_tool = get_tool_for_obj(obs_obj)

    # XDL 문자열
    cur_xdl = generate_xdl_string(cur_op, cur_targets)
    next_xdl = generate_xdl_string(next_op, next_targets) if next_op else "None"

    # candidate grids
    target_grids = [obj_grid_map[o] for o in all_target_objs]
    categories = generate_candidate_grids(
        occupied, obs_grid, target_grids, next_op
    )
    if count_candidates(categories) < 3:
        return None

    # 목적지 결정
    target_grid = determine_target_grid(obs_obj, next_op, next_targets,
                                         categories)
    if not target_grid:
        return None

    # obstacle info
    obstacle_info = f"{obs_obj} at {obs_grid} (blocking)"

    return {
        "instruction": {
            "current_xdl": cur_xdl,
            "next_xdl": next_xdl,
            "obstacle_info": obstacle_info,
            "candidate_grids": format_candidate_grids(categories),
        },
        "output": f"{main_tool},True,{aux_tool},{target_grid}"
    }


def generate_not_blocking_entry(cur_op, cur_targets, next_op, next_targets,
                                  all_target_objs, obs_obj):
    """장애물이 있지만 blocking하지 않는 경우 데이터 생성"""
    available_grids = list(PLACEABLE_GRIDS)
    random.shuffle(available_grids)

    obj_grid_map = {}
    for obj in all_target_objs:
        if not available_grids:
            return None
        g = available_grids.pop(0)
        obj_grid_map[obj] = g

    # 장애물을 비인접 그리드에 배치
    primary_target_grid = obj_grid_map[cur_targets[0]]
    adj_grids = set(ADJACENCY.get(primary_target_grid, []))
    target_grids_set = set(obj_grid_map.values())

    non_adj = [g for g in available_grids
               if g not in adj_grids and g not in target_grids_set and g != "G2"]
    if not non_adj:
        return None

    obs_grid = random.choice(non_adj)
    obj_grid_map[obs_obj] = obs_grid
    occupied = set(obj_grid_map.values())

    # main_tool
    main_tool = get_main_tool(cur_op, cur_targets[0] if cur_targets else None)

    # XDL 문자열
    cur_xdl = generate_xdl_string(cur_op, cur_targets)
    next_xdl = generate_xdl_string(next_op, next_targets) if next_op else "None"

    # candidate grids
    target_grids = [obj_grid_map[o] for o in all_target_objs]
    categories = generate_candidate_grids(
        occupied, None, target_grids, next_op
    )
    if count_candidates(categories) < 3:
        return None

    # obstacle info (not blocking이지만 환경에 존재)
    obstacle_info = f"{obs_obj} at {obs_grid}"

    return {
        "instruction": {
            "current_xdl": cur_xdl,
            "next_xdl": next_xdl,
            "obstacle_info": obstacle_info,
            "candidate_grids": format_candidate_grids(categories),
        },
        "output": f"{main_tool},False,None,None"
    }


# =============================================================================
# 8. 균등 분포 샘플링
# =============================================================================

def balanced_sample(data_by_category, target_total=3000):
    """
    카테고리 간 균등 분포로 샘플링
    목표: 재배치 필요 50% / 불필요 50%
          재배치 필요 중 문맥 인식 50% / 무관 50%
    """
    no_rearrange = (
        data_by_category.get("no_rearrange_no_obstacle", []) +
        data_by_category.get("no_rearrange_not_blocking", [])
    )
    rearrange_context = data_by_category.get("rearrange_context", [])
    rearrange_irrelevant = data_by_category.get("rearrange_irrelevant", [])

    half = target_total // 2
    quarter = half // 2

    # 샘플링 (부족하면 최대한)
    sampled_no = random.sample(no_rearrange, min(half, len(no_rearrange)))
    sampled_ctx = random.sample(rearrange_context, min(quarter, len(rearrange_context)))
    sampled_irr = random.sample(rearrange_irrelevant, min(quarter, len(rearrange_irrelevant)))

    result = sampled_no + sampled_ctx + sampled_irr
    random.shuffle(result)
    return result


# =============================================================================
# 9. 검증
# =============================================================================

def validate_entry(entry):
    """생성된 데이터의 무결성 검증"""
    inst = entry["instruction"]
    output = entry["output"]
    tokens = output.split(",")

    # 출력 4토큰 확인
    assert len(tokens) == 4, f"Output must be 4 tokens: {output}"

    main_tool, need_rearrange, aux_tool, target_grid = tokens

    # 허용 값 확인
    assert main_tool in ["dh3", "ag95", "vgc10"], f"Invalid main_tool: {main_tool}"
    assert need_rearrange in ["True", "False"], f"Invalid need_rearrange: {need_rearrange}"
    assert aux_tool in ["dh3", "ag95", "vgc10", "None"], f"Invalid aux_tool: {aux_tool}"

    # 재배치 불필요 시 aux_tool과 target_grid는 None
    if need_rearrange == "False":
        assert aux_tool == "None", f"aux_tool must be None when no rearrange: {output}"
        assert target_grid == "None", f"target_grid must be None when no rearrange: {output}"

    # 재배치 필요 시 aux_tool과 target_grid는 None이면 안 됨
    if need_rearrange == "True":
        assert aux_tool != "None", f"aux_tool required when rearrange: {output}"
        assert target_grid != "None", f"target_grid required when rearrange: {output}"

    # target_grid가 candidate grids에 포함되는지 확인
    if target_grid != "None":
        assert target_grid in inst["candidate_grids"], \
            f"target_grid {target_grid} not in candidates: {inst['candidate_grids']}"

    return True


# =============================================================================
# 10. 메인 실행
# =============================================================================

def main():
    print("Generating ActionReasoner training data...")
    all_data, data_by_category = generate_data()

    print(f"\n--- Raw generation stats ---")
    print(f"Total raw entries: {len(all_data)}")
    for cat, entries in data_by_category.items():
        print(f"  {cat}: {len(entries)}")

    # 균등 샘플링
    TARGET_TOTAL = 3000
    sampled = balanced_sample(data_by_category, TARGET_TOTAL)
    print(f"\n--- After balanced sampling ---")
    print(f"Total sampled: {len(sampled)}")

    # 분포 확인
    rearrange_count = sum(1 for e in sampled if ",True," in e["output"])
    no_rearrange_count = sum(1 for e in sampled if ",False," in e["output"])
    print(f"  Rearrange needed: {rearrange_count} ({100*rearrange_count/len(sampled):.1f}%)")
    print(f"  No rearrange: {no_rearrange_count} ({100*no_rearrange_count/len(sampled):.1f}%)")

    # 검증
    print("\nValidating...")
    valid_count = 0
    invalid_count = 0
    for entry in sampled:
        try:
            validate_entry(entry)
            valid_count += 1
        except AssertionError as e:
            invalid_count += 1
            if invalid_count <= 5:
                print(f"  INVALID: {e}")

    print(f"  Valid: {valid_count}, Invalid: {invalid_count}")

    # 저장
    output_path = "/home/claude/action_reasoner_dataset.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in sampled:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDataset saved to: {output_path}")
    print(f"Total entries: {len(sampled)}")


if __name__ == "__main__":
    main()