# Environment Relationship Diagram

This diagram captures the high-level relationship in the HMARL maritime environment.

```mermaid
flowchart LR
    subgraph Forecasting
      M[Medium-term forecast\ndefault 5 days\n(configurable medium_horizon_days)]
      S[Short-term forecast\\n6-24 hours]
    end

    subgraph CoordinatorLayer
      C[Fleet Coordinator]
    end

    subgraph OperationalLayer
      V[Vessel Agents]
      P[Port Agents]
    end

    subgraph Env
      E[Maritime Environment\\nTransit + Queues + Docks + Emissions]
      R[Reward + Metrics Engine]
    end

    M --> C
    C -->|Strategic directives| V
    S --> V
    S --> P
    V -->|Arrival requests + speed actions| P
    V -->|Movement actions| E
    P -->|Service rate + dock actions| E
    E -->|State observations| C
    E -->|State observations| V
    E -->|State observations| P
    E --> R
    R -->|R_C| C
    R -->|R_V| V
    R -->|R_P| P
```

## Environment prep implications

1. Keep one source of truth for dynamics (`step_vessels`, `step_ports`).
2. Keep observation builders deterministic per tick.
3. Separate forecast generation from policy logic.
4. Keep reward and metrics computation centralized and testable.

## Position model note

Current `position_nm` update is a simplifying MVP assumption: 1D progress along
a route edge (rectilinear in route-distance space). Keep it for skeleton-level
validation, then upgrade to graph/waypoint movement in the research phase.
