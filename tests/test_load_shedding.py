"""POST /load-shedding 的端到端测试。

测试用树（与 test_evaluate 相同）：

source --s1(0.1Ω, 14A)--> a --s2(0.2Ω, 10A)--> b
                           |
                           +--s3(0.05Ω, 15A)--> c

负载（序号即 loads 数组下标，从 0 开始）：
- 0: a, current=2
- 1: b, current=5
- 2: c, current=8

基线：段电流 s1=15（超额定 14A）, s2=5, s3=8；节点电压 a=46.5, b=45.5, c=46.1
切除负载 0（a 的 2A）后 s1=13，全树安全，切除电流总和最小（2A）。
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**overrides):
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 14.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 15.0},
        ],
        "loads": [
            {"node": "a", "current": 2.0},
            {"node": "b", "current": 5.0},
            {"node": "c", "current": 8.0},
        ],
        "sheddable_loads": [0, 1, 2],
    }
    payload.update(overrides)
    return payload


def by_id(items, key):
    return {item[key]: item for item in items}


# ---------- 有解 ----------


def test_selects_minimum_shed_current():
    """s1 超额定 1A：切除 0 号 2A 即可恢复，切除电流总和最小。"""
    resp = client.post("/load-shedding", json=base_payload())
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "solved"
    assert data["shed_loads"] == [0]
    assert data["shed_count"] == 1
    assert data["shed_current"] == pytest.approx(2.0)
    assert data["sheddable_loads"] == [0, 1, 2]

    ev = data["evaluation"]
    assert ev["ok"] is True
    assert ev["over_current_segments"] == []
    assert ev["under_voltage_nodes"] == []
    assert ev["total_source_current"] == pytest.approx(13.0)
    currents = by_id(ev["segments"], "segment_id")
    assert currents["s1"]["current"] == pytest.approx(13.0)
    assert currents["s2"]["current"] == pytest.approx(5.0)
    assert currents["s3"]["current"] == pytest.approx(8.0)
    nodes = by_id(ev["nodes"], "node")
    assert nodes["a"]["load_current"] == pytest.approx(0.0)
    assert nodes["a"]["voltage"] == pytest.approx(46.7)
    assert nodes["b"]["voltage"] == pytest.approx(45.7)


def test_excluding_cheapest_load_forces_larger_shed():
    """0 号不可切除时须切除 1 号 5A（s1 由 15 降到 10）。"""
    resp = client.post("/load-shedding", json=base_payload(sheddable_loads=[1, 2]))
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "solved"
    assert data["shed_loads"] == [1]
    assert data["shed_current"] == pytest.approx(5.0)
    ev = data["evaluation"]
    assert ev["ok"] is True
    assert ev["total_source_current"] == pytest.approx(10.0)
    currents = by_id(ev["segments"], "segment_id")
    assert currents["s1"]["current"] == pytest.approx(10.0)
    assert currents["s2"]["current"] == pytest.approx(0.0)
    assert currents["s3"]["current"] == pytest.approx(8.0)


def test_already_safe_selects_empty_set():
    """原网已安全时选空集，校核结果与 /evaluate 一致。"""
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 20.0
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "solved"
    assert data["shed_loads"] == []
    assert data["shed_count"] == 0
    assert data["shed_current"] == pytest.approx(0.0)
    ev = data["evaluation"]
    assert ev["ok"] is True
    assert ev["total_source_current"] == pytest.approx(15.0)


def test_empty_sheddable_list_when_safe():
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 20.0
    payload["sheddable_loads"] = []
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == []


def test_unsorted_input_indices_normalized():
    """响应中的可切除序号归一化为升序。"""
    resp = client.post("/load-shedding", json=base_payload(sheddable_loads=[2, 0, 1]))
    assert resp.status_code == 200
    data = resp.json()
    assert data["sheddable_loads"] == [0, 1, 2]
    assert data["shed_loads"] == [0]


def test_shed_entire_load_only_node_aggregation():
    """同节点多条负载分别计数，被切除者整条移除后节点负载为剩余之和。"""
    payload = base_payload()
    payload["loads"] = [
        {"node": "a", "current": 2.0},
        {"node": "b", "current": 3.0},
        {"node": "b", "current": 2.0},
        {"node": "c", "current": 8.0},
    ]
    # 基线 s1 = 15；只允许切除 1 号（b 的 3A）：s1=12 安全，b 剩余 2A
    payload["sheddable_loads"] = [1]
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["shed_loads"] == [1]
    assert data["shed_current"] == pytest.approx(3.0)
    ev = data["evaluation"]
    assert ev["ok"] is True
    nodes = by_id(ev["nodes"], "node")
    assert nodes["b"]["load_current"] == pytest.approx(2.0)
    currents = by_id(ev["segments"], "segment_id")
    assert currents["s2"]["current"] == pytest.approx(2.0)


# ---------- 择优平局 ----------


def test_tie_lexicographic_smallest_index_list():
    """三条负载各 3A、s1 额定 8A（基线 9A）：切任意一条都安全，切除电流与条数
    相同，取升序序号字典序最小者 [0]。"""
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.0, "rated_current": 8.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.0, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.0, "rated_current": 15.0},
        ],
        "loads": [
            {"node": "a", "current": 3.0},
            {"node": "b", "current": 3.0},
            {"node": "c", "current": 3.0},
        ],
        "sheddable_loads": [0, 1, 2],
    }
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [0]
    assert data["shed_current"] == pytest.approx(3.0)


def test_tie_fewer_shed_count_wins():
    """切除电流总和相同时条数少者胜：一条 4A 与两条各 2A 均可恢复时选 [0]。"""
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.0, "rated_current": 4.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.0, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.0, "rated_current": 15.0},
        ],
        "loads": [
            {"node": "a", "current": 4.0},
            {"node": "b", "current": 2.0},
            {"node": "c", "current": 2.0},
        ],
        "sheddable_loads": [0, 1, 2],
    }
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [0]
    assert data["shed_count"] == 1


def test_under_voltage_drives_shedding():
    """欠压同样必须消除：s1 电阻 0.5Ω，基线 a=40.5/b=39.5（b 欠压）；
    在 [1,2] 中切除 1 号 5A 后 a=43、b=42 恢复，比切除 8A 更省。"""
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 20.0
    payload["segments"][0]["resistance"] = 0.5
    payload["sheddable_loads"] = [1, 2]
    # 基线：s1=15 -> a=40.5, b=39.5 欠压；切 1 号 5A -> s1=10, a=43, b=42
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [1]
    ev = data["evaluation"]
    assert ev["ok"] is True
    assert ev["under_voltage_nodes"] == []


# ---------- 无解 ----------


def test_infeasible_when_no_shedding_allowed():
    resp = client.post("/load-shedding", json=base_payload(sheddable_loads=[]))
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "infeasible"
    assert data["shed_loads"] == []
    assert data["shed_count"] == 0
    assert data["shed_current"] == pytest.approx(0.0)
    ev = data["evaluation"]
    assert ev["ok"] is False
    assert [v["segment_id"] for v in ev["over_current_segments"]] == ["s1"]


def test_infeasible_when_violation_upstream_of_unsheddable_load():
    """a 点 9A 不可切除，s1(1Ω) 使 a=39V 欠压；切除下游 b 的 2A 后 s1 仍承载
    a 点 9A，a 电压恰好仍为 39V，欠压无法消除。"""
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 1.0, "rated_current": 50.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.0, "rated_current": 50.0},
        ],
        "loads": [
            {"node": "a", "current": 9.0},
            {"node": "b", "current": 2.0},
        ],
        "sheddable_loads": [1],
    }
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "infeasible"
    assert data["shed_loads"] == [1]
    assert data["shed_current"] == pytest.approx(2.0)
    ev = data["evaluation"]
    assert ev["ok"] is False
    # s2 电阻为零，b 与 a 同为 39V，均欠压
    assert [v["node"] for v in ev["under_voltage_nodes"]] == ["a", "b"]
    assert ev["total_source_current"] == pytest.approx(9.0)


# ---------- 入参校验 ----------


def test_duplicate_index_rejected():
    resp = client.post("/load-shedding", json=base_payload(sheddable_loads=[1, 1]))
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("重复" in d["message"] for d in error["details"])


def test_out_of_range_index_rejected():
    resp = client.post("/load-shedding", json=base_payload(sheddable_loads=[3]))
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("越界" in d["message"] for d in error["details"])


def test_negative_index_rejected():
    resp = client.post("/load-shedding", json=base_payload(sheddable_loads=[-1]))
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error"
    assert any("可切除负载序号" in d["message"] for d in error["details"])


def test_too_many_sheddable_loads_rejected():
    resp = client.post("/load-shedding", json=base_payload(sheddable_loads=list(range(17))))
    # loads 只有 3 条，超长校验先于越界触发 422
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert any("16" in d["message"] for d in error["details"])


def test_missing_sheddable_loads_rejected():
    payload = base_payload()
    del payload["sheddable_loads"]
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 422
    assert any("可切除负载序号列表" in d["message"] for d in resp.json()["error"]["details"])


def test_invalid_tree_still_rejected():
    payload = base_payload()
    payload["segments"].append(
        {"parent": "x", "child": "y", "resistance": 0.1, "rated_current": 10.0}
    )
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"


def test_sixteen_sheddable_loads_accepted_and_enumerated():
    """恰好 16 条可切除负载允许提交且全组合被枚举（2^16=65536）。"""
    segments = [
        {"id": "s1", "parent": "source", "child": "a", "resistance": 0.0, "rated_current": 50.0}
    ]
    loads = [{"node": "a", "current": 1.0} for _ in range(16)]
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": segments,
        "loads": loads,
        "sheddable_loads": list(range(16)),
    }
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # 基线 16A 超额定 15A？额定 50 安全 -> 空集
    assert data["status"] == "solved"
    assert data["shed_loads"] == []


def test_node_min_voltage_override_kept_when_other_load_shed():
    """节点级最低允许电压随节点生效：切除无覆盖的负载后覆盖仍保留。"""
    payload = base_payload()
    payload["loads"][0]["min_voltage"] = 47.0  # a 阈值 47V，基线 a=46.5 欠压且 s1 超额定
    # 切 0 号 2A 后 a=46.7 仍欠压；切 1 号 5A 后 s1=10、a 恰为 47.0（触限可行）
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [1]
    ev = data["evaluation"]
    assert ev["ok"] is True
    nodes = by_id(ev["nodes"], "node")
    assert nodes["a"]["min_voltage"] == pytest.approx(47.0)
    assert nodes["a"]["voltage"] == pytest.approx(47.0)
    assert nodes["a"]["under_voltage"] is False
