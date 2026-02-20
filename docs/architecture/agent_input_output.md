# Agent Input / Output Diagram

## Per-agent I/O schema

| Agent | Inputs | Outputs | Cadence (target) |
|------|--------|---------|------------------|
| Fleet Coordinator | Medium forecast, fleet summaries, cumulative emissions | Destination directive, departure window, emission budget | Every 12 steps |
| Vessel Agent | Coordinator directive, short forecast, vessel local state | Target speed, arrival-slot request | Every step |
| Port Agent | Queue/dock state, incoming requests, short forecast | Service rate, request acceptance, dock allocation | Every step |

## Dataflow diagram

```mermaid
flowchart TD
    CO[Coordinator Obs]
    VA[Vessel Obs]
    PA[Port Obs]

    C[Fleet Coordinator Policy]
    V[Vessel Policy]
    P[Port Policy]

    D[Directive]
    ACTV[Vessel Actions]
    ACTP[Port Actions]

    CO --> C --> D
    D --> V
    VA --> V --> ACTV
    ACTV --> P
    PA --> P --> ACTP
```

## Current code mapping

1. Coordinator policy:
   `hmarl_mvp/policies.py::fleet_coordinator_policy`
2. Vessel policy:
   `hmarl_mvp/policies.py::vessel_policy`
3. Port policy:
   `hmarl_mvp/policies.py::port_policy`
4. Observation builders:
   `hmarl_mvp/env.py::_get_observations`

