"""
AtlasResQ AI Ã¢â‚¬â€ Comprehensive Part 2 Backend Test Suite

Covers:
  1. Database Initialization & Seeding
  2. User Authentication (Register, Duplicate Check, Login, JWT, Me, Logout)
  3. Role-Based Access Control (Public vs Official vs Admin vs Unauthenticated)
  4. Priority Scoring Engine (Formulas, Levels, Uncertainty Safety Rule)
  5. Image Analysis Pipeline (Upload, Real Inference, Storage, Record, History, Ownership)
  6. Verification Workflow & Dynamic Reprioritization
  7. Command Center Dashboard Summary
  8. System Health & Real Model Status

Run:
    $env:PYTHONIOENCODING='utf-8'; python backend/tests/test_part2.py
"""

import sys
import os
import json
import io
import time
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image

# Add backend directory to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND = _ROOT / "backend"
sys.path.insert(0, str(_BACKEND))

from app.main import app
from app.database.database import SessionLocal, init_db
from app.database.seed import seed_database
from app.models.user import User, UserRole
from app.models.region import Region, PriorityLevel, VerificationStatus
from app.priority.priority_engine import priority_engine

client = TestClient(app)

PASS = "Ã¢Å“â€¦ PASS"
FAIL = "Ã¢ÂÅ’ FAIL"
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


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â 1. DATABASE & SEED INITIALIZATION Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")

def test_db_init_and_seed():
    init_db()
    seed_database()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == "admin@atlasresq.gov").first()
        official = db.query(User).filter(User.email == "official@atlasresq.gov").first()
        user = db.query(User).filter(User.email == "user@atlasresq.org").first()
        assert admin is not None and admin.role == UserRole.ADMIN, "Admin seed missing"
        assert official is not None and official.role == UserRole.OFFICIAL, "Official seed missing"
        assert user is not None and user.role == UserRole.PUBLIC_USER, "Public user seed missing"

        # Check seeded regions
        regions = db.query(Region).all()
        assert len(regions) >= 6, f"Expected at least 6 seeded regions, found {len(regions)}"
    finally:
        db.close()

run_test("Database tables and seed data created", test_db_init_and_seed)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â 2. USER AUTHENTICATION & JWT Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")

_public_token = None
_official_token = None
_admin_token = None
# Use timestamp+pid for unique email across repeated test runs against shared DB
_registered_user_email = f"testuser_{os.getpid()}_{int(time.time())}@example.com"

def test_register_public_user():
    """Register a new public user. Cleans up any pre-existing user with same email first."""
    # Cleanup any leftover from previous run (shared DB)
    from app.database.database import SessionLocal
    from app.models.user import User as UserModel
    db = SessionLocal()
    try:
        existing = db.query(UserModel).filter(UserModel.email == _registered_user_email).first()
        if existing:
            db.delete(existing)
            db.commit()
    finally:
        db.close()

    res = client.post(
        "/api/auth/register",
        json={
            "email": _registered_user_email,
            "password": "Password123!",
            "full_name": "Test Public Citizen",
            "organization": "Volunteer Network",
        },
    )
    assert res.status_code == 201, f"Register failed: {res.text}"
    data = res.json()
    assert data["email"] == _registered_user_email
    assert data["role"] == "PUBLIC_USER"
    assert "password_hash" not in data, "Password hash must never be returned in API"

def test_duplicate_email_rejected():
    res = client.post(
        "/api/auth/register",
        json={
            "email": _registered_user_email,
            "password": "Password123!",
            "full_name": "Duplicate User",
        },
    )
    assert res.status_code == 400, "Duplicate email should return 400 Bad Request"

def test_login_valid_credentials():
    global _public_token
    res = client.post(
        "/api/auth/login",
        json={"email": _registered_user_email, "password": "Password123!"},
    )
    assert res.status_code == 200, f"Login failed: {res.text}"
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == _registered_user_email
    _public_token = data["access_token"]

def test_login_invalid_password():
    res = client.post(
        "/api/auth/login",
        json={"email": _registered_user_email, "password": "WrongPassword999"},
    )
    assert res.status_code == 401, "Invalid password should return 401 Unauthorized"

def test_auth_me_endpoint():
    assert _public_token is not None
    res = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {_public_token}"},
    )
    assert res.status_code == 200, f"/me failed: {res.text}"
    data = res.json()
    assert data["email"] == _registered_user_email

def test_stateless_logout():
    assert _public_token is not None
    res = client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {_public_token}"},
    )
    assert res.status_code == 200
    assert "Logout successful" in res.json()["message"]

run_test("User registration (POST /api/auth/register)", test_register_public_user)
run_test("Duplicate email rejection (400)", test_duplicate_email_rejected)
run_test("User login & JWT issuance (POST /api/auth/login)", test_login_valid_credentials)
run_test("Invalid password rejection (401)", test_login_invalid_password)
run_test("Current user profile (GET /api/auth/me)", test_auth_me_endpoint)
run_test("Stateless logout (POST /api/auth/logout)", test_stateless_logout)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â 3. ROLE-BASED ACCESS CONTROL (RBAC) Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")

def test_login_official_and_admin():
    global _official_token, _admin_token
    res_off = client.post("/api/auth/login", json={"email": "official@atlasresq.gov", "password": "Official@123456"})
    assert res_off.status_code == 200, f"Official login failed: {res_off.text}"
    _official_token = res_off.json()["access_token"]

    res_adm = client.post("/api/auth/login", json={"email": "admin@atlasresq.gov", "password": "Admin@123456"})
    assert res_adm.status_code == 200, f"Admin login failed: {res_adm.text}"
    _admin_token = res_adm.json()["access_token"]

def test_unauthenticated_request_rejected():
    res = client.get("/api/dashboard/summary")
    assert res.status_code == 401, "Unauthenticated request to protected endpoint must return 401"

def test_public_user_forbidden_from_official_endpoints():
    res = client.get("/api/dashboard/summary", headers={"Authorization": f"Bearer {_public_token}"})
    assert res.status_code == 403, "Public user accessing official endpoint must return 403"

    res_q = client.get("/api/verification/queue", headers={"Authorization": f"Bearer {_public_token}"})
    assert res_q.status_code == 403, "Public user accessing verification queue must return 403"

def test_public_user_forbidden_from_admin_endpoints():
    res = client.get("/api/users", headers={"Authorization": f"Bearer {_public_token}"})
    assert res.status_code == 403, "Public user accessing admin endpoint must return 403"

def test_official_user_allowed_on_official_endpoints():
    res = client.get("/api/dashboard/summary", headers={"Authorization": f"Bearer {_official_token}"})
    assert res.status_code == 200, "Official user should access dashboard summary"

    res_q = client.get("/api/verification/queue", headers={"Authorization": f"Bearer {_official_token}"})
    assert res_q.status_code == 200, "Official user should access verification queue"

def test_admin_user_allowed_on_admin_endpoints():
    res = client.get("/api/users", headers={"Authorization": f"Bearer {_admin_token}"})
    assert res.status_code == 200, "Admin user should access user management"
    assert "users" in res.json()

run_test("Login seeded Official and Admin accounts", test_login_official_and_admin)
run_test("Unauthenticated requests return 401", test_unauthenticated_request_rejected)
run_test("Public user forbidden from official endpoints (403)", test_public_user_forbidden_from_official_endpoints)
run_test("Public user forbidden from admin endpoints (403)", test_public_user_forbidden_from_admin_endpoints)
run_test("Official user allowed on official endpoints (200)", test_official_user_allowed_on_official_endpoints)
run_test("Admin user allowed on admin endpoints (200)", test_admin_user_allowed_on_admin_endpoints)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â 4. PRIORITY ENGINE LOGIC & SAFETY Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")

def test_priority_engine_high_damage():
    # Low calibrated_probability -> high damage
    score, level, breakdown, rec = priority_engine.calculate_priority(
        calibrated_probability=0.05,
        tta_uncertainty=0.01,
        verification_status=VerificationStatus.UNVERIFIED,
    )
    assert score >= 80.0, f"Expected CRITICAL/HIGH score, got {score}"
    assert level in (PriorityLevel.CRITICAL, PriorityLevel.HIGH)

def test_priority_engine_low_damage():
    # High calibrated_probability -> no damage
    score, level, breakdown, rec = priority_engine.calculate_priority(
        calibrated_probability=0.95,
        tta_uncertainty=0.01,
        verification_status=VerificationStatus.UNVERIFIED,
    )
    assert score < 35.0, f"Expected LOW score for no-damage, got {score}"
    assert level == PriorityLevel.LOW

def test_priority_uncertainty_safety_rule():
    # High damage + High uncertainty must NOT blindly maximize score, but flag for human review
    score, level, breakdown, rec = priority_engine.calculate_priority(
        calibrated_probability=0.10,
        tta_uncertainty=0.12,  # high variance
        verification_status=VerificationStatus.UNVERIFIED,
    )
    assert breakdown["uncertainty_flag"] == "UNCERTAIN_DAMAGE_VERIFICATION_RECOMMENDED"
    assert "Verification Recommended" in rec

def test_priority_verification_override():
    # Confirmed damage overrides to floor >= 85
    score_conf, level_conf, _, _ = priority_engine.calculate_priority(
        calibrated_probability=0.50,
        tta_uncertainty=0.05,
        verification_status=VerificationStatus.CONFIRMED_DAMAGE,
    )
    assert score_conf >= 85.0
    assert level_conf == PriorityLevel.CRITICAL

    # Confirmed no damage overrides to ceiling <= 10
    score_safe, level_safe, _, _ = priority_engine.calculate_priority(
        calibrated_probability=0.10,
        tta_uncertainty=0.02,
        verification_status=VerificationStatus.NO_DAMAGE,
    )
    assert score_safe <= 10.0
    assert level_safe == PriorityLevel.LOW

run_test("Priority engine: high damage scoring", test_priority_engine_high_damage)
run_test("Priority engine: low damage scoring", test_priority_engine_low_damage)
run_test("Uncertainty safety rule: flags review for uncertain outputs", test_priority_uncertainty_safety_rule)
run_test("Verification status overrides priority appropriately", test_priority_verification_override)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â 5. IMAGE ANALYSIS & INFERENCE API Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")

_test_analysis_id = None

def test_real_image_inference_upload():
    global _test_analysis_id
    test_img = _ROOT / "gradcam_true_positive.png"
    assert test_img.exists(), f"Test image {test_img} not found"

    with open(test_img, "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/api/analysis/infer",
        headers={"Authorization": f"Bearer {_public_token}"},
        files={"file": ("damage_test.png", img_bytes, "image/png")},
        data={"region_code": "A1", "gradcam": "true", "mode": "LIVE"},
    )
    assert res.status_code == 201, f"Inference upload failed: {res.text}"
    data = res.json()
    assert "id" in data
    assert data["prediction"] in ("Damage", "No Damage")
    assert 0.0 <= data["damage_probability"] <= 1.0
    assert 0.0 <= data["calibrated_probability"] <= 1.0
    assert data["gradcam_available"] is True
    assert data["mode"] == "LIVE"
    assert "AI-estimated visible damage" in data["disclaimer"]
    _test_analysis_id = data["id"]

def test_invalid_file_rejected():
    res = client.post(
        "/api/analysis/infer",
        headers={"Authorization": f"Bearer {_public_token}"},
        files={"file": ("fake.txt", b"not an image", "text/plain")},
    )
    assert res.status_code == 400, "Text file upload must return 400"

def test_analysis_history_and_ownership():
    assert _test_analysis_id is not None
    # Public user can see own analysis
    res = client.get(
        "/api/analysis",
        headers={"Authorization": f"Bearer {_public_token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 1

    # Public user can get own analysis by ID
    res_single = client.get(
        f"/api/analysis/{_test_analysis_id}",
        headers={"Authorization": f"Bearer {_public_token}"},
    )
    assert res_single.status_code == 200
    assert res_single.json()["id"] == _test_analysis_id

    # Official user can also view the analysis
    res_off = client.get(
        f"/api/analysis/{_test_analysis_id}",
        headers={"Authorization": f"Bearer {_official_token}"},
    )
    assert res_off.status_code == 200

run_test("Real image inference upload (POST /api/analysis/infer)", test_real_image_inference_upload)
run_test("Invalid file format rejected (400)", test_invalid_file_rejected)
run_test("Analysis history and access control (GET /api/analysis)", test_analysis_history_and_ownership)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â 6. REGIONAL INTELLIGENCE & VERIFICATION WORKFLOW Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")

_test_region_id = None
_test_verification_id = None

def test_list_regions_and_priority():
    global _test_region_id
    res = client.get(
        "/api/priority/regions",
        headers={"Authorization": f"Bearer {_official_token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 6
    # Pick region A1 explicitly for deterministic verification test
    a1_region = next(r for r in data["regions"] if r["region_code"] == "A1")
    _test_region_id = a1_region["id"]

def test_verification_queue():
    res = client.get(
        "/api/verification/queue",
        headers={"Authorization": f"Bearer {_official_token}"},
    )
    assert res.status_code == 200
    queue = res.json()
    assert len(queue) >= 1

def test_create_official_verification():
    global _test_verification_id
    assert _test_region_id is not None
    res = client.post(
        "/api/verification",
        headers={"Authorization": f"Bearer {_official_token}"},
        json={
            "region_id": _test_region_id,
            "decision": "CONFIRMED_DAMAGE",
            "comment": "Drone survey confirmed structural roof collapse in Sector A1.",
        },
    )
    assert res.status_code == 201, f"Create verification failed: {res.text}"
    data = res.json()
    assert data["decision"] == "CONFIRMED_DAMAGE"
    _test_verification_id = data["id"]

def test_dynamic_priority_recalculation():
    # Check that the region priority was updated to confirmed critical floor >= 85
    res = client.get(
        "/api/priority/regions/A1",
        headers={"Authorization": f"Bearer {_official_token}"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["verification_status"] == "CONFIRMED_DAMAGE"
    assert data["priority_score"] >= 85.0

run_test("List operational grid regions (GET /api/priority/regions)", test_list_regions_and_priority)
run_test("Verification queue for officials (GET /api/verification/queue)", test_verification_queue)
run_test("Submit official verification (POST /api/verification)", test_create_official_verification)
run_test("Dynamic priority recalculation on verification", test_dynamic_priority_recalculation)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â 7. COMMAND CENTER DASHBOARD & SYSTEM HEALTH Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")

def test_dashboard_summary():
    res = client.get(
        "/api/dashboard/summary",
        headers={"Authorization": f"Bearer {_official_token}"},
    )
    assert res.status_code == 200, f"Dashboard summary failed: {res.text}"
    data = res.json()
    assert "total_analyses" in data
    assert "damage_regions" in data
    assert "high_priority_regions" in data
    assert "verified_regions" in data
    assert data["model_status"]["loaded"] is True

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["database_connected"] is True
    assert data["model_loaded"] is True

def test_model_status_endpoint():
    res = client.get("/model/status")
    assert res.status_code == 200
    data = res.json()
    assert data["loaded"] is True
    assert data["model_name"] == "EfficientNet-B0"
    assert data["threshold"] == 0.37499999999999994
    assert data["calibration_method"] == "temperature_scaling"

run_test("Command Center summary (GET /api/dashboard/summary)", test_dashboard_summary)
run_test("System health status (GET /health)", test_health_endpoint)
run_test("Real model status (GET /model/status)", test_model_status_endpoint)


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
print("\nÃ¢â€ÂÃ¢â€ÂÃ¢â€Â RESULTS SUMMARY Ã¢â€ÂÃ¢â€ÂÃ¢â€Â")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)
print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}\n")

for name, ok, err in results:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {name}")
    if err:
        print(f"       Ã¢â€â€Ã¢â€â‚¬ {err}")

if failed > 0:
    print(f"\nÃ¢Å¡Â Ã¯Â¸Â  {failed} run_test(s) failed.")
    sys.exit(1)
else:
    print(f"\nÃ¢Å“â€¦ All {total} Part 2 backend tests passed.")


