# CASE (Student Version) — **Aurelia Retail: AI-Driven Decisions Under Constraints**

## 1. Context (Feb 2026)
Aurelia Retail is a mid-sized omnichannel retailer operating:
- 38 physical stores
- 1 central warehouse + 2 regional cross-docks
- a fast-growing e-commerce channel (same-day options in two cities)

Over the last 12 months, Aurelia has faced compounding pressures:
- **SLA breaches** (late deliveries) increased from 6% to 14%
- **stockouts** increased in high-margin categories
- a rise in **customer churn** for the subscription loyalty plan
- an emerging **fraud pattern** in online checkout
- operational costs rising (overtime + expedited shipping)

The CEO asks the Analytics team to build an “AI decision cockpit” that improves service while controlling cost and risk.

## 2. Your Role
You are the lead Business Analytics team. You must propose an AI roadmap and deliver an initial prototype that:
1) formulates key operational problems precisely,
2) chooses appropriate algorithmic approaches (search/optimization vs ML),
3) sets evaluation metrics and governance practices,
4) is explainable to operations and finance.

## 3. The Decision Domains (4 workstreams)
A. **Warehouse picking routes** (internal routing)
B. **Store staffing / shift scheduling**
C. **Demand forecasting for replenishment**
D. **Fraud detection + churn risk prioritization**

## 4. Exhibit 1 — Operational Constraints (excerpt)
**Scheduling**
- Each shift must have exactly 1 staff member assigned.
- No employee can work two shifts on the same day.
- Max 5 shifts per week per employee.
- Preferences exist (soft constraints): avoid Sundays, avoid PM→next-day AM.

**Picking**
- Warehouse is modeled as a grid with blocked aisles (temporary closures).
- Pick list changes daily; congestion varies by time of day.
- Objective: complete all picks with minimum distance/time, avoid congestion if possible.

**Forecasting**
- The business uses weekly ordering; holding cost is material.
- Promotions cause demand spikes; stockouts are expensive.

**Risk**
- Fraud is rare but costly; false negatives are expensive.
- Churn interventions have a budget; precision matters.

## 5. Exhibit 2 — KPI definitions (what “good” means)
- On-time delivery (OTD) target: 95% within SLA
- Stockout rate target: < 2.5%
- Overtime cost reduction: -10%
- Loyalty churn reduction: -15% relative
- Fraud loss reduction: -20%

## 6. Exhibit 3 — Data snapshots (simplified)
Synthetic datasets are provided in the repository:
- `data/churn.csv`
- `data/demand.csv`
- `data/fraud.csv`

## 7. Assignment Questions
1) Choose **two** domains (A–D). For each, decide: is it primarily **search/optimization** or **ML** (or a hybrid)? Justify.
2) For one domain, write a full formulation (S, A, T, G, C, R). Identify hard vs soft constraints.
3) Propose at least two alternative representations and explain trade-offs (granularity, state size, cost definition).
4) Define an evaluation plan: metrics, baselines, and failure modes (e.g., drift, imbalance, fairness concerns).
5) Design a governance “minimum viable” checklist: reproducibility, traceability, monitoring, communication.

## 8. Closing
In one page, propose an implementation roadmap for the next 12 weeks (phases, deliverables, risks).
