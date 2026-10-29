"""
AtlasResQ AI — Test Suite Part 4: Disaster Warning & Early Preparedness Layer
"""

import sys
import os
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.database.database import init_db
from app.database.seed import seed_database
from app.models.warning import WarningSeverity, WarningType, WarningStatus
from app.models.rescue_team import TeamStatus

# ANSI styling
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"

client = TestClient(app)

_official_token = None
_admin_token = None
_public_token = None

def run_test(name: str, test_func):
    try:
        test_func()
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        print(f"  {FAIL}  {name}\n        Error: {e}")
        return False

def test_auth_setup():
    global _official_token, _admin_token, _public_token
    init_db()
    seed_database()

    # Login official
    res = client.post("/api/auth/login", json={"email": "official@atlasresq.gov", "password": "Official@123456"})
    assert res.status_code == 200, f"Official login failed: {res.text}"
    _official_token = res.json()["access_token"]

    # Login admin
    res = client.post("/api/auth/login", json={"email": "admin@atlasresq.gov", "password": "Admin@123456"})
    assert res.status_code == 200, f"Admin login failed: {res.text}"
    _admin_token = res.json()["access_token"]

    # Login public
    res = client.post("/api/auth/login", json={"email": "user@atlasresq.org", "password": "User@123456"})
    assert res.status_code == 200, f"Public login failed: {res.text}"
    _public_token = res.json()["access_token"]

def test_list_warnings():
    res = client.get("/api/warnings", headers={"Authorization": f"Bearer {_public_token}"})
    assert res.status_code == 200, f"List warnings failed: {res.text}"
    data = res.json()
    assert "total" in data
    assert "active_count" in data
    assert data["total"] >= 1
    assert data["active_count"] >= 1
    w0 = data["warnings"][0]
    assert w0["is_demo"] is True
    assert "source" in w0

def test_get_preparedness_matrix():
    res = client.get("/api/warnings/preparedness", headers={"Authorization": f"Bearer {_official_token}"})
    assert res.status_code == 200, f"Preparedness matrix failed: {res.text}"
    data = res.json()
    assert "total_sectors" in data
    assert data["total_sectors"] >= 6
    assert "items" in data
    item0 = data["items"][0]
    assert "preparedness_score" in item0
    assert "preparedness_status" in item0
    assert item0["preparedness_status"] in ["READY", "PREPARE", "AT_RISK", "URGENT_PREPARATION"]

def test_create_warning():
    import uuid
    code = f"WARN-TEST-{uuid.uuid4().hex[:6].upper()}"
    res = client.post(
        "/api/warnings",
        headers={"Authorization": f"Bearer {_official_token}"},
        json={
            "warning_code": code,
            "warning_type": "FLOOD",
            "title": "Flash Flood Watch for Industrial Grid",
            "source": "NDMA SACHET",
            "severity": "HIGH",
            "status": "ACTIVE",
            "affected_region_codes": "B1",
            "affected_district": "Industrial Sector",
            "description": "Test flash flood advisory.",
            "is_demo": True,
        }
    )
    assert res.status_code == 201, f"Create warning failed: {res.text}"
    data = res.json()
    assert data["warning_code"] == code

def test_pre_position_rescue_team():
    res = client.post(
        "/api/rescue-teams/TEAM-USAR-01/pre-position?region_code=A1",
        headers={"Authorization": f"Bearer {_official_token}"},
    )
    assert res.status_code == 200, f"Pre-position team failed: {res.text}"
    data = res.json()
    assert data["status"] == "PRE_POSITIONED"
    assert data["assigned_region_code"] == "A1"

def test_agent_warning_tools():
    res = client.post(
        "/api/agent/chat",
        headers={"Authorization": f"Bearer {_official_token}"},
        json={"message": "Which sectors have active warnings and require immediate preparation?"}
    )
    assert res.status_code == 200, f"Agent warning chat failed: {res.text}"
    data = res.json()
    assert "reply" in data
    assert len(data["reply"]) > 10

if __name__ == "__main__":
    print(f"\n{BOLD}=== ATLASRESQ DISASTER WARNING & PREPAREDNESS TESTS ==={RESET}\n")
    tests = [
        ("Auth setup for test client", test_auth_setup),
        ("List active disaster warnings API (GET /api/warnings)", test_list_warnings),
        ("Preparedness matrix API (GET /api/warnings/preparedness)", test_get_preparedness_matrix),
        ("Create warning advisory (POST /api/warnings)", test_create_warning),
        ("Pre-position rescue team (POST /api/rescue-teams/{code}/pre-position)", test_pre_position_rescue_team),
        ("Response Assistant warning tool execution", test_agent_warning_tools),
    ]

    passed = 0
    for name, func in tests:
        if run_test(name, func):
            passed += 1

    print(f"\n{BOLD}Completed {len(tests)} warning tests. Passed: {passed}, Failed: {len(tests) - passed}{RESET}\n")
    if passed < len(tests):
        sys.exit(1)
