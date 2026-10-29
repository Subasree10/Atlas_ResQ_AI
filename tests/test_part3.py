"""
AtlasResQ AI — Command Center Upgrade Test Suite (Part 3)
Tests:
  1. Rescue Teams CRUD and Dynamic Assignment API
  2. Safe Location Recommendation, Safety Rules & Routing
  3. Auditable Incident Timeline API
  4. Incident Briefing Summary & Readiness Score
  5. Response Intelligence Assistant Tool Execution
"""

import sys
import os
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = _ROOT / "backend"
sys.path.insert(0, str(_BACKEND))

from app.main import app
from app.database.database import SessionLocal, init_db
from app.database.seed import seed_database
from app.models.user import User, UserRole
from app.models.region import Region
from app.models.rescue_team import RescueTeam, TeamStatus
from app.models.safe_location import SafeLocation

client = TestClient(app)

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = []


def run_test(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        msg = str(e)
        print(f"  {FAIL}  {name}")
        print(f"          {msg}")
        results.append((name, False, msg))


print("\n=== ATLASRESQ COMMAND CENTER UPGRADE TESTS ===")

# Setup Auth Tokens
_official_token = None


def test_auth_setup():
    global _official_token
    init_db()
    seed_database()
    resp = client.post("/api/auth/login", json={"email": "official@atlasresq.gov", "password": "Official@123456"})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    _official_token = resp.json()["access_token"]


run_test("Auth setup for official user", test_auth_setup)


def test_list_and_recommend_rescue_teams():
    headers = {"Authorization": f"Bearer {_official_token}"}
    resp = client.get("/api/rescue-teams", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 5
    assert "available_teams" in data

    rec_resp = client.get("/api/rescue-teams/recommend/A1", headers=headers)
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["region_code"] == "A1"
    assert len(rec_data["recommended_teams"]) > 0


run_test("Rescue teams listing and recommendation engine", test_list_and_recommend_rescue_teams)


def test_assign_and_unassign_rescue_team():
    headers = {"Authorization": f"Bearer {_official_token}"}
    # List available teams
    teams_resp = client.get("/api/rescue-teams?status_filter=AVAILABLE", headers=headers)
    teams = teams_resp.json()["teams"]
    assert len(teams) > 0, "No available rescue teams for testing"
    team_id = teams[0]["id"]

    # Assign team to A2
    assign_resp = client.post("/api/rescue-teams/assign", headers=headers, json={
        "team_id": team_id,
        "region_code": "A2",
        "task_notes": "Deploy to commercial district triage point"
    })
    assert assign_resp.status_code == 200, f"Assign failed: {assign_resp.text}"
    assigned = assign_resp.json()
    assert assigned["status"] == "DEPLOYED"
    assert assigned["assigned_region_code"] == "A2"

    # Verify region DB updated
    reg_resp = client.get("/api/priority/regions", headers=headers)
    reg_a2 = [r for r in reg_resp.json()["regions"] if r["region_code"] == "A2"][0]
    assert reg_a2["assigned_team_code"] == assigned["team_code"]

    # Unassign team
    unassign_resp = client.post(f"/api/rescue-teams/unassign/{team_id}", headers=headers)
    assert unassign_resp.status_code == 200
    assert unassign_resp.json()["status"] == "AVAILABLE"


run_test("Dynamic team assignment and unassignment lifecycle", test_assign_and_unassign_rescue_team)


def test_safe_location_recommendation_and_safety_rules():
    headers = {"Authorization": f"Bearer {_official_token}"}
    # Valid region code recommendation
    resp = client.get("/api/safe-locations/recommend?region_code=A1", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_verified_data"] is True
    assert data["safety_message"] == "Recommended Safe Location"
    assert len(data["recommendations"]) > 0
    assert "route_waypoints" in data["recommendations"][0]

    # Test missing coordinates rule
    no_loc_resp = client.get("/api/safe-locations/recommend", headers=headers)
    assert no_loc_resp.status_code == 200
    no_loc_data = no_loc_resp.json()
    assert no_loc_data["has_verified_data"] is False
    assert "cannot be reliably determined" in no_loc_data["safety_message"]


run_test("Safe location recommendation and safety rule enforcement", test_safe_location_recommendation_and_safety_rules)


def test_timeline_and_summary_apis():
    headers = {"Authorization": f"Bearer {_official_token}"}
    # Timeline
    t_resp = client.get("/api/timeline", headers=headers)
    assert t_resp.status_code == 200
    assert t_resp.json()["total"] > 0

    # Incident brief summary
    s_resp = client.get("/api/summary/incident-brief", headers=headers)
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    assert "operational_readiness_score" in s_data
    assert "recommended_actions" in s_data
    assert s_data["mode_indicator"] in ["LIVE INFERENCE", "DEMO DATA"]


run_test("Incident timeline and briefing summary APIs", test_timeline_and_summary_apis)


def test_response_intelligence_agent():
    headers = {"Authorization": f"Bearer {_official_token}"}
    q_resp = client.post("/api/agent/chat", headers=headers, json={"message": "Summarize the current incident."})
    assert q_resp.status_code == 200, f"Agent chat error: {q_resp.text}"
    q_data = q_resp.json()
    assert q_data["status"] == "success"
    assert "Incident Briefing" in q_data["reply"]


run_test("Response Intelligence Assistant tool execution", test_response_intelligence_agent)


if __name__ == "__main__":
    failed = [r for r in results if not r[1]]
    print(f"\nCompleted {len(results)} tests. Failed: {len(failed)}")
    if failed:
        sys.exit(1)
