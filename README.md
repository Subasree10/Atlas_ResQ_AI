# AtlasResQ AI
> **AI-Powered Disaster Damage Intelligence & Response Prioritization**

This is the production-ready emergency-response decision-support platform designed to ingest post-disaster satellite imagery, analyze building damage patterns using temperature-calibrated deep learning, estimate neural uncertainty with Test-Time Augmentation (TTA), explain decisions with Grad-CAM heatmaps, and prioritize relief staging through a human-in-the-loop verification pipeline.

---

## 🚀 Quick Start Instructions

### 1. Environment Configuration
Copy the default environment template:
```bash
cp backend/.env.example backend/.env
```

### 2. Backend Setup
1. **Navigate to the Backend**:
   ```bash
   cd backend
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure PyTorch CPU/CUDA version is loaded correctly.*
3. **Database Initialization & Seeding**:
   The database initializes automatically on API gateway lifespan startup. Default seed users:
   - **Admin**: `admin@atlasresq.gov` (Password: `admin123`)
   - **Official**: `official@atlasresq.gov` (Password: `official123`)
   - **Public User**: `user@atlasresq.org` (Password: `user123`)
4. **Start the API Server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### 3. Frontend Setup
1. **Navigate to the Frontend**:
   ```bash
   cd frontend
   ```
2. **Install Dependencies**:
   ```bash
   npm install
   ```
3. **Run the Development Server**:
   ```bash
   npm run dev
   ```
   Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🧪 Verification & Test Suite

Run the full automated test suite containing 52/52 passing verification checkpoints:

```bash
# From workspace root
python backend/tests/test_part1.py
python backend/tests/test_part2.py
```

### Build Verification
Verify frontend production build:
```bash
cd frontend
npm run build
```

---

## 🛡️ Operational Guardrails & Scientific Disclaimers
1. **Data Limitations**: Relative grid sectors (`A1`–`B3`) are utilized in place of GPS coordinates to prevent fabricated location mappings.
2. **Safety Notice**: *“AI outputs identify visible structural changes associated with building damage patterns. They do not determine survivors, casualties, occupancy, or confirmed rescue requirements.”*
3. **Uncertainty Alert Rule**: High model uncertainty (TTA variance $\ge$ 8%) triggers an explicit *"Human Verification Recommended"* warning to prevent incorrect priority staging decisions.
