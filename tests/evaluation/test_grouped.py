"""Tests for the grouped breakdowns and the Markdown report (synthetic data only)."""

from __future__ import annotations

from holdon.data.wayup import LabeledEvent, participant_of
from holdon.evaluation.grouped import (
    by_experience,
    by_limb_type,
    by_route,
    experience_groups,
    load_experience,
    occlusion_frame_recall,
)
from holdon.evaluation.report import breakdown_report
from holdon.events import ContactEvent


def ev(vid, start, end, hold="1", limb="left_hand", score=1.0):
    return ContactEvent(vid, limb, hold, start, end, score)


def test_hands_feet_filter_both_sides():
    gt = {"p1_green": [ev("p1_green", 0, 9), ev("p1_green", 0, 9, limb="left_foot")]}
    pred = {"p1_green": [ev("p1_green", 0, 9, limb="left_foot"), ev("p1_green", 50, 59, limb="left_foot")]}
    rows = by_limb_type(pred, gt, [0.5])
    assert rows["hands"][0.5]["recall"] == 0.0 and rows["hands"][0.5]["precision"] == 0.0
    assert rows["feet"][0.5]["precision"] == 0.5 and rows["feet"][0.5]["recall"] == 1.0


def test_route_macro():
    gt = {v: [ev(v, 0, 9)] for v in ("p1_green", "p2a_green", "p1_orange")}
    pred = {"p1_green": [ev("p1_green", 0, 9)], "p1_orange": [ev("p1_orange", 0, 9)]}
    rows = by_route(pred, gt, [0.5])
    assert list(rows) == ["green", "orange"]
    assert rows["green"][0.5]["f1"] == 0.5 and rows["orange"][0.5]["f1"] == 1.0


def test_occlusion_frame_recall_pools_frames():
    gt = {"v_green": [LabeledEvent(ev("v_green", 0, 9), ((0, 3),)),          # 4 occluded, 6 visible
                      LabeledEvent(ev("v_green", 20, 29, hold="2"), ())]}      # 10 visible
    pred = {"v_green": [ev("v_green", 2, 5),               # covers occluded 2-3, visible 4-5
                        ev("v_green", 20, 29, hold="9")]}  # wrong hold: covers nothing
    fr = occlusion_frame_recall(pred, gt)
    assert (fr["occluded"].frames, fr["occluded"].covered) == (4, 2)
    assert (fr["visible"].frames, fr["visible"].covered) == (16, 2)
    assert occlusion_frame_recall(pred, gt, limbs=("left_foot",))["visible"].recall == 0.0


def test_experience_groups_merge_small_categories(tmp_path):
    form = tmp_path / "form.csv"
    form.write_text(
        "Participant ID,Age,For how long have you been climbing?,notes\n"
        '1,x,1-2 years,\n2,x,more than 10 years,"multi\nline"\n3,x,less than 1 year,\n'
        "4,x,1-2 years,\n5,x,more than 10 years,\n", encoding="utf-8")
    exp = load_experience(form)
    groups = experience_groups(exp, min_participants=2)
    assert groups[3] == groups[1] == "less than 1 year + 1-2 years"
    assert groups[2] == "more than 10 years"
    assert experience_groups({1: "1-2 years", 2: "5-10 years"}, 2) == {
        1: "1-2 years + 5-10 years", 2: "1-2 years + 5-10 years"}


def test_by_experience_reports_group_averages_and_sizes():
    gt = {v: [ev(v, 0, 9)] for v in ("p1_green", "p2a_green", "p2b_green", "p3_green")}
    pred = {"p1_green": [ev("p1_green", 0, 9)], "p2a_green": [ev("p2a_green", 0, 9)]}
    out = by_experience(pred, gt, {1: "A", 2: "B", 3: "A"}, participant_of, [0.5])
    assert out["A"][:2] == (2, 2) and out["A"][2][0.5]["f1"] == 0.5
    assert out["B"][:2] == (1, 2) and out["B"][2][0.5]["f1"] == 0.5


def test_breakdown_report_sections():
    gt = {v: [LabeledEvent(ev(v, 0, 9), ((0, 4),))] for v in ("p1_green", "p1_orange")}
    pred = {"p1_green": [ev("p1_green", 0, 9)]}
    text = "\n".join(breakdown_report(pred, gt, [0.0, 0.5], title="t", experience_groups={1: "A"}))
    for heading in ("All limbs", "Route", "Occluded", "experience"):
        assert heading in text
    assert "| all | 0.500 / 0.500 / **0.500**" in text
