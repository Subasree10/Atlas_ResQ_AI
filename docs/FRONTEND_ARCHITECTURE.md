# AtlasResQ AI — Frontend Architecture (Future)

> Not built in Part 1. This outlines the intended approach for Parts 2–3.

## Stack
- React 18+ with TypeScript
- Vite (build tool)
- Tailwind CSS (utility-first styling)
- shadcn/ui (component library)
- Lucide React (icons)
- Framer Motion (animations)
- TanStack Query (data fetching)
- Recharts (charts)
- React Leaflet (maps, when georeferenced data available)

## Design Standard
Professional Emergency Operations Center + Satellite Intelligence Platform.

NOT: generic Bootstrap dashboard, basic college project, or template clone.

Must feature:
- Strong visual hierarchy and typography
- Satellite imagery as core visual element
- Meaningful status indicators
- Refined animations and loading states
- Responsive desktop/tablet/mobile
- Accessible interactions

## Intended Page Structure
- `/` — Landing / Public portal
- `/login` — Authentication
- `/public/assess` — Public damage assessment
- `/official/dashboard` — Command centre overview
- `/official/assess` — Full assessment with all pipeline outputs
- `/official/priority` — Priority map / regional ranking
- `/official/verify` — Human verification workflow
- `/official/agent` — ResQ Agent interface
- `/admin` — System administration
