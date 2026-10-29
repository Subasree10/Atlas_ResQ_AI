# AtlasResQ AI — Architecture

## System Overview

AtlasResQ AI is a disaster-response decision-support system that uses trained satellite imagery classification to provide preliminary damage assessments.

```
POST-DISASTER SATELLITE IMAGE
          ↓
    IMAGE PREPROCESSING
    (RGB, Resize 255, CenterCrop 224, Normalize ImageNet)
          ↓
    IMAGE TILING / REGIONAL ANALYSIS  [Future]
          ↓
    EFFICIENTNET-B0
    (Trained on Kaggle "Satellite Images of Hurricane Damage")
          ↓
    DAMAGE PROBABILITY
    (sigmoid of raw logit)
          ↓
    CALIBRATION
    (Temperature Scaling, T=1.2597)
          ↓
    TTA UNCERTAINTY
    (4 augmented variants → mean + std)
          ↓
    GRAD-CAM
    (Attention visualisation via features[-1])
          ↓
    PRIORITY ENGINE            [Future]
          ↓
    PRIORITY MAP               [Future]
          ↓
    RANKED REGIONS             [Future]
          ↓
    HUMAN VERIFICATION         [Future]
          ↓
    DYNAMIC RE-PRIORITIZATION  [Future]
          ↓
    RESQ AGENT                 [Future]
          ↓
    RESPONSE PLANNING          [Future]
```

## Technology Stack

### Backend
- Python 3.10+
- FastAPI (REST API)
- PyTorch 2.x + Torchvision (inference)
- Pillow (image processing)
- Pydantic v2 (validation)
- SQLAlchemy (future database)
- Uvicorn (ASGI server)

### Frontend (Part 3)
- React + TypeScript
- Vite
- Tailwind CSS
- shadcn/ui, Lucide, Framer Motion, TanStack Query, Recharts, React Leaflet

## Geographic Data Policy

The training dataset (Kaggle "Satellite Images of Hurricane Damage") does not embed geographic coordinates. Therefore:
- The system uses **relative region grid notation** (A1, A2, B1, etc.)
- Latitude/longitude are **NEVER invented**
- Future integration with georeferenced satellite imagery will provide real coordinates

## User Roles (Future)

### Official Users (Command Centre)
- Full priority intelligence
- Regional analysis dashboard
- Human verification workflow
- ResQ Agent interface
- Response planning

### Public Users (Simple Assessment)
- Upload satellite imagery
- Receive damage likelihood
- Community observation reporting

## Security Architecture (Future)
- JWT authentication
- Role-based access control
- Password hashing (bcrypt)
- File validation + size limits (20 MB)
- CORS controls
- Audit logging
- Secret management via environment variables
