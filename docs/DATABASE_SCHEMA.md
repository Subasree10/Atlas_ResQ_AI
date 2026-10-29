# AtlasResQ AI — Database Schema (Future)

> This document outlines the planned database schema. Not yet implemented in Part 1.

## Tables

### `users`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| email | VARCHAR(255) | UNIQUE |
| hashed_password | TEXT | bcrypt |
| role | ENUM('public','official','admin') | |
| created_at | TIMESTAMP | |

### `assessments`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| user_id | UUID | FK → users |
| image_path | TEXT | Stored image reference |
| prediction | VARCHAR(20) | "Damage" / "No Damage" |
| damage_probability | FLOAT | |
| calibrated_probability | FLOAT | |
| tta_mean_probability | FLOAT | |
| tta_uncertainty | FLOAT | |
| confidence_level | VARCHAR(10) | HIGH/MEDIUM/LOW |
| verified | BOOLEAN | Human verification status |
| verified_by | UUID | FK → users (nullable) |
| region_id | VARCHAR(10) | Grid reference (e.g. "A1") |
| created_at | TIMESTAMP | |

### `regions`
| Column | Type | Notes |
|---|---|---|
| id | VARCHAR(10) | PK (e.g. "A1") |
| priority_score | FLOAT | Computed priority |
| assessment_count | INT | |
| damage_ratio | FLOAT | |
| updated_at | TIMESTAMP | |

### `audit_log`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| user_id | UUID | FK → users |
| action | VARCHAR(50) | |
| details | JSONB | |
| created_at | TIMESTAMP | |
