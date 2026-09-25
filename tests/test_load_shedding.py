"""POST /load-shedding 的端到端测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**overrides):
    """与 test_evaluate 相同的三层小树：

    source --s1(0.1Ω)--> a --s2(0.2Ω)--> b
                          |
                          +--s3(0.05Ω)--> c

    负载：a=2A（序号 0）, b=5A（序号 1）, c=8A（序号 2）
    安全基线：段电流 s1=15, s2=5, s3=8；节点电压 a=46.5, b=45.5, c=46.1
    """
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
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


# ---------- already_safe ----------


def test_already_safe_network_returns_empty_set():
    resp = client.post("/load-shedding", json=base_payload())
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "already_safe"
    assert data["shed_loads"] == []
    assert data["shed_count"] == 0
    assert data["shed_current"] == pytest.approx(0.0)
    assert data["shed_details"] == []
    assert data["result"]["ok"] is True
    assert data["result"]["total_source_current"] == pytest.approx(15.0)


def test_already_safe_without_sheddable_field():
    payload = base_payload()
    del payload["sheddable_loads"]
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_safe"


# ---------- solved ----------


def test_shed_minimum_current_for_over_current():
    # s1 额定降为 14A（实际 15A）：仅切 a(2A) 即可，优于切 b(5A)/c(8A)
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 14.0
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "solved"
    assert data["shed_loads"] == [0]
    assert data["shed_count"] == 1
    assert data["shed_current"] == pytest.approx(2.0)
    assert data["shed_details"] == [{"index": 0, "node": "a", "current": pytest.approx(2.0)}]

    result = data["result"]
    assert result["ok"] is True
    assert result["over_current_segments"] == []
    assert result["under_voltage_nodes"] == []
    assert result["total_source_current"] == pytest.approx(13.0)
    segments = by_id(result["segments"], "segment_id")
    assert segments["s1"]["current"] == pytest.approx(13.0)
    assert segments["s2"]["current"] == pytest.approx(5.0)
    assert segments["s3"]["current"] == pytest.approx(8.0)
    assert all(s["over_rated"] is False for s in result["segments"])

    nodes = by_id(result["nodes"], "node")
    assert nodes["a"]["load_current"] == pytest.approx(0.0)  # a 的负载已整条切除
    assert nodes["b"]["load_current"] == pytest.approx(5.0)
    assert nodes["c"]["load_current"] == pytest.approx(8.0)
    assert nodes["a"]["voltage"] == pytest.approx(46.7)
    assert nodes["b"]["voltage"] == pytest.approx(45.7)
    assert nodes["c"]["voltage"] == pytest.approx(46.3)


def test_shed_branch_load_for_branch_over_current():
    # s2 额定 4A（实际 5A）：必须切 b(5A)，切 a/c 无效
    payload = base_payload()
    payload["segments"][1]["rated_current"] = 4.0
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [1]
    assert data["shed_current"] == pytest.approx(5.0)
    result = data["result"]
    segments = by_id(result["segments"], "segment_id")
    assert segments["s2"]["current"] == pytest.approx(0.0)
    assert segments["s1"]["current"] == pytest.approx(10.0)


def test_shed_for_under_voltage():
    # 电源 43V、s2 电阻 0.5Ω：b=39V 欠压；切 b(5A) 后 b 恢复至 42V，切 c(8A) 对 b 无效
    payload = base_payload(
        source_voltage=43.0,
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 30.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.5, "rated_current": 30.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 30.0},
        ],
    )
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [1]
    assert data["shed_current"] == pytest.approx(5.0)
    assert data["result"]["ok"] is True
    nodes = by_id(data["result"]["nodes"], "node")
    assert nodes["b"]["voltage"] == pytest.approx(42.0)
    assert nodes["b"]["load_current"] == pytest.approx(0.0)


def test_tie_prefer_fewer_shed_count():
    # 全部负载挂在 b 点：s1/s2 各承载 8A（额定 4A），需切除至少 4A
    # 单切 0 号 4A（1 条）与切 1、2 号各 2A（2 条）切除电流并列 -> 条数少者胜
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.01, "rated_current": 4.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.01, "rated_current": 4.0},
        ],
        "loads": [
            {"node": "b", "current": 4.0},
            {"node": "b", "current": 2.0},
            {"node": "b", "current": 2.0},
        ],
        "sheddable_loads": [0, 1, 2],
    }
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [0]
    assert data["shed_count"] == 1
    assert data["shed_current"] == pytest.approx(4.0)


def test_tie_same_current_and_count_lexicographic():
    # b、c 各 5A，单切任一即可消除 s1 越限，电流与条数并列 -> 序号小者 [1]
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 10.0
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [1]
    assert data["shed_current"] == pytest.approx(5.0)


def test_tie_multi_element_combo_lexicographic():
    # 四条负载各 3A（b、c 各两条），s1 额定 8A（实际 12A）须切 2 条（6A）：
    # 任意两条均可，电流 / 条数并列时取序号升序元组字典序最小者 [0,1]
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.01, "rated_current": 8.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.01, "rated_current": 30.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.01, "rated_current": 30.0},
        ],
        "loads": [
            {"node": "b", "current": 3.0},
            {"node": "b", "current": 3.0},
            {"node": "c", "current": 3.0},
            {"node": "c", "current": 3.0},
        ],
        "sheddable_loads": [0, 1, 2, 3],
    }
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [0, 1]
    assert data["shed_count"] == 2
    assert data["shed_current"] == pytest.approx(6.0)


def test_exact_rated_and_exact_threshold_are_safe():
    # s1 恰好承载等于额定的电流：原网即安全（EPS 容差，触限不算越限）
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 15.0
    resp = client.post("/load-shedding", json=payload)
    assert resp.json()["status"] == "already_safe"


def test_sheddable_subset_excludes_non_candidates():
    # s1 越限，但 a(0) 不在可切除列表：只能从 b/c 中选最小可行者 b(5A)
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 14.0
    payload["sheddable_loads"] = [1, 2]
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [1]


def test_sheddable_order_does_not_affect_result():
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 14.0
    payload["sheddable_loads"] = [2, 0, 1]
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["shed_loads"] == [0]  # 返回值恒为升序


def test_zero_current_load_never_selected():
    payload = base_payload()
    payload["loads"].append({"node": "a", "current": 0.0})  # 序号 3
    payload["segments"][0]["rated_current"] = 14.0
    payload["sheddable_loads"] = [0, 1, 2, 3]
    resp = client.post("/load-shedding", json=payload)
    data = resp.json()
    assert data["status"] == "solved"
    assert data["shed_loads"] == [0]  # 不会为消越限切零电流负载


# ---------- infeasible ----------


def test_infeasible_when_violation_cannot_be_cleared():
    # s2 额定 4A（b 支路过流），但 b(1) 不可切除
    payload = base_payload()
    payload["segments"][1]["rated_current"] = 4.0
    payload["sheddable_loads"] = [0, 2]
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "infeasible"
    assert data["shed_loads"] is None
    assert data["shed_count"] is None
    assert data["shed_current"] is None
    assert data["shed_details"] is None
    assert data["result"] is None


def test_empty_sheddable_with_violation_is_infeasible():
    payload = base_payload()
    payload["segments"][0]["rated_current"] = 14.0
    payload["sheddable_loads"] = []
    resp = client.post("/load-shedding", json=payload)
    assert resp.json()["status"] == "infeasible"


# ---------- 输入校验 ----------


def test_duplicate_index_rejected():
    payload = base_payload(sheddable_loads=[1, 1])
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("序号 1 重复" in d["message"] for d in error["details"])


def test_out_of_range_index_rejected():
    payload = base_payload(sheddable_loads=[3])
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 400
    assert any("越界" in d["message"] for d in resp.json()["error"]["details"])


def test_negative_index_rejected():
    payload = base_payload(sheddable_loads=[-1])
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("越界" in d["message"] for d in error["details"])


def test_non_integer_index_rejected():
    payload = base_payload(sheddable_loads=[1.5])
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 422


def test_too_many_sheddable_rejected():
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 100.0},
        ],
        "loads": [{"node": "a", "current": 1.0} for _ in range(17)],
        "sheddable_loads": list(range(17)),
    }
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 422
    assert any("16" in d["message"] for d in resp.json()["error"]["details"])


def test_sixteen_sheddable_accepted():
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 100.0},
        ],
        "loads": [{"node": "a", "current": 1.0} for _ in range(16)],
        "sheddable_loads": list(range(16)),
    }
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_safe"


def test_existing_tree_validation_still_applies():
    payload = base_payload()
    payload["loads"].append({"node": "ghost", "current": 1.0})
    payload["sheddable_loads"] = [0]
    resp = client.post("/load-shedding", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"
