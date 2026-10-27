# Enterprise split: design (written before any candidate was built)

## Question

Can federated ULTRA give a data-limited cybersecurity client a meaningful gain over training on
its own data? "Meaningful" is the project's fixed criterion: a mean target-test MRR gain of at
least +0.03 over target-only fine-tuning, with 95% CIs whose lower bounds are above 0. Test
metrics are never used to choose data, partitions, hyperparameters or checkpoints.

## Why a split of one domain

Enterprise→ICS and Enterprise→Mobile showed no transfer. Part of the reason is that the domains
differ in schema and content: ICS's main relation has no Enterprise counterpart. Splitting
Enterprise between two clients removes that mismatch. The clients still differ in the one way
real organisations do: **each tracks different threat actors**, so each knows a different set of
actor behaviours, software and techniques.

**Scenario.** Two threat-intelligence teams model the same ATT&CK technique space:
- A **source** team tracks its own portfolio of groups and campaigns.
- A **target** team tracks a different portfolio and has documented only part of what its actors
  do. It holds 10%, 25% or 100% of its facts.

Neither team shares raw intelligence; they share only model weights (FedAvg). Whether the
split creates anything worth exchanging is measured, not assumed (Phase 2 below).

## Input

- Bundle: Enterprise ATT&CK v19.2 STIX (`datasets/cybersecurity/mitre/enterprise/v19.2/enterprise-attack-19.2.json`,
  sha256 `dc1639ca…c3d8f4`, the same file the frozen `fedultra_fresh` split was built from).
- Facts: the frozen builder's `extract()`, which keeps active objects only and the relations
  uses, mitigates, detects, subtechnique-of and attributed-to. That gives 21,103 facts and 2,495 entities.
- No fact is added. Facts may only be withheld.

## Partition algorithm (v1)

1. **Actors and units.**
   - *Actors* are the intrusion-sets and campaigns that head at least one fact (232).
   - Two actors are *associated* if:
     - (a) a STIX `attributed-to` relationship links them;
     - (b) one's description links to the other's ATT&CK page (`attack.mitre.org/groups/Gxxxx` or
       `/campaigns/Cxxxx`); or
     - (c) they share an alias (case-insensitive).
   - *Units* are the connected components of this association graph.
   - A unit's *actor facts* are all facts headed by its actors: actor uses technique, actor uses
     software, and campaign attributed-to group. Both ends of an attribution lie in one unit by
     construction.
   - Keeping associated actors together stops overlapping real-world actors (different STIX
     objects with cross-references) from landing in different clients.
2. **Unit assignment.** This is the only step where candidates differ (see Candidates).
3. **The source holds its units' actor facts; the target holds its units' actor facts.**
4. **Target evaluation split** (seed 20261203). This is done before step 5, so it does not depend
   on how non-actor facts are assigned.
   - Shuffle the target's actor facts with relation `uses` (sorted, then seeded).
   - Take 10% as validation and 10% as test.
   - Move back to training any validation or test fact whose head or tail appears in no remaining
     target actor training fact.
   - `attributed-to` facts are training only.
5. **Ownership of non-actor facts, by actual STIX usage.**
   - *Usage* u_c(e) is the number of client c's actor **training** facts whose tail is e.
     Target validation and test facts are excluded.
   - **Software facts** (software uses technique): all facts headed by software S go to
     argmax_c u_c(S).
   - **Reference facts** go to the owner of the technique they describe, argmax_c u_c(T):
     - for `mitigates` and `detects`, the tail technique;
     - for `subtechnique-of`, the head (the sub-technique).
   - Ties, including entities that no actor uses, are broken by a seeded coin keyed by the
     entity's STIX id (seed 20261202).
   - Rationale: a team documents the software and techniques its own actors use.
   - Every fact goes to exactly one client.
6. **Source.** All source facts are training facts. The source is never evaluated.
7. **Target scarcity** (nested; seed 20261204).
   - For every head entity h of the target's training facts, shuffle h's facts once (keyed by
     h) and keep the first max(1, round(f·n_h)), for f ∈ {0.10, 0.25, 1.00}.
   - Removed facts go to `hidden.txt`: true facts the client does not observe. They are filtered
     at evaluation and never trained on.
   - Validation and test are the same at every f. Every head entity keeps at least one fact. So
     the data-limited client still has a thin profile of every actor and software it tracks,
     instead of losing small actors entirely.

## Candidates

Both candidates use the same steps 1 and 3–7.

- **C-rand.** Sort units by actor-fact count (descending, ties by smallest STIX id) and pair
  them consecutively. A seeded coin (seed 20261201) sends one unit of each pair to the target and
  the other to the source; an odd unit out gets its own coin. These are two portfolios drawn from
  the same actor population.
- **C-time.** Date each unit by the earliest STIX `created` timestamp of its actors (ties by
  smallest STIX id). Assign units to the source in date order until it holds at least 50% of
  actor facts; the rest go to the target. This is an established team with long-documented actors
  and a newer team tracking recently documented ones.

## Leakage definitions (all must be zero)

- **L1:** no source training fact equals any target fact (train, hidden, valid or test), directly
  or inverse-equivalently. A fact (t, r′, h) for any r′ is inverse-equivalent to (h, r, t).
- **L2:** target valid, test and train ∪ hidden are pairwise disjoint, directly and
  inverse-equivalently.
- **L3:** no source training fact connects the two entities of any target validation or test
  fact, in any relation or direction.
- **L4:** no target actor appears in the source graph, and no source actor appears in the target graph.

Semantic overlap that these checks cannot see is reported separately (see Phase 2, item 2).

## Phase 2 viability metrics (per candidate, at f = 0.10, 0.25, 1.00)

1. Entities, relations, triples and relation frequencies per client.
2. Shared entities by type. Exact, inverse-equivalent and entity-pair overlaps (L1–L4).
   Remaining semantic overlap is listed separately.
3. Multi-hop compositions in each client's training graph:
   - distinct relation paths of length 2 and 3 with at least one instance;
   - rules body ⇒ `uses` with support ≥ 2;
   - entropy of rule support.
4. Source support for real held-out target queries, on **validation only**:
   - (a) the share of target validation facts whose endpoints a path of length ≤ 3 connects in
     the target training graph;
   - (b) the share of target validation facts with at least one such path that is also a rule
     with source support ≥ 2;
   - (c) validation rule MRR with target-only, source-only and pooled rule statistics. **Pooled −
     own is the transfer-opportunity proxy.** It is the k ≤ 3 path-rule test used in the audit
     (`transfer_conditions/src/explore_enrichment.py`).
5. All of the above at each f.
6. Validation and test counts, and the share of validation and test facts whose head and tail
   each keep at least one target training edge at each f.

**Proxy calibration** (descriptive). The same proxy is computed for the earlier cells whose
FedAvg − local results are known:
- Study S (8 cells, test Δ);
- Mobile (3 cells, test Δ);
- ICS (3 cells, validation Δ).

This shows how far the proxy tracks real ULTRA gains. It does not change the rules below.

## Hard constraints (a candidate must pass all)

- **V1:** L1–L4 are all zero.
- **V2:** at every f, the target's training relations equal the source's, and every validation
  and test relation is in target training.
- **V3:** validation and test each have at least 250 facts.
- **V4:** at every f, at least 80% of validation facts and at least 80% of test facts have both
  endpoints with a target training edge.

## Selection and go/no-go (fixed now)

- **Candidate.** Among candidates that pass V1–V4, take the one with the larger mean proxy over
  f ∈ {0.10, 0.25}.
- **Primary fraction.** For that candidate, take the f ∈ {0.10, 0.25} with the larger proxy;
  ties go to 0.10.
- **Go** if:
  - (i) the selected candidate passes V1–V4;
  - (ii) the proxy at the primary fraction is above 0, so the source's evidence improves the
    target's validation ranking (a plausible transfer opportunity);
  - (iii) own-rule validation MRR at the primary fraction is below own-rule MRR at f = 1.00, so
    scarcity hurts (need);
  - (iv) the automated leakage and coverage tests pass on the written files.
- **No-go** otherwise. The study then stops with an explanation, and no fallback to synthetic
  data or a re-cut of the partition is made.

Seeds for the confirmatory runs and the analysis are fixed in the preregistration, written after
Phase 2 and before any target test metric exists.

## Amendment 1 (2026-10-26, written before any candidate viability number was read)

**Reason.** The proxy calibration (`results/proxy_calibration.json`) shows that go rules (ii) and
(iii), and the proxy-based choices of candidate and primary fraction, are not valid screens for
ULTRA.
- In the two cells with the largest real federated gains, the proxy was about 0: Study S shared
  had −0.019 at n = 13 (actual Δ +0.100) and +0.000 at n = 41 (actual Δ +0.118).
- Own-rule MRR was higher at n = 13 than at n = 408 (0.401 vs 0.376), so rule (iii) would also
  have rejected them.
- A rule learner can pick the right rule from a handful of facts. ULTRA's fine-tuning cannot, so
  rule-based "need" does not measure ULTRA's need.

What does separate the cell families is **availability**: how well the source's own rules rank
the target's held-out facts, relative to the target's own rules, on the target's complete graph.

| Family | Source-only / own rule MRR at full data | FedAvg − local |
|---|---|---|
| Study S shared | 0.3788 / 0.3762 = 1.01 | +0.10 to +0.12 when scarce |
| Mobile | 0.1024 / 0.1408 = 0.73 | ≈ 0 |
| ICS | 0.1014 / 0.3159 = 0.32 | ≈ 0 |
| Study S disjoint | 0.0764 / 0.3762 = 0.20 | ≤ 0 |

**Replacement rules.** These replace "Selection and go/no-go" above. The proxy and own-rule
numbers are still computed and reported, but only descriptively.
- **Availability** A = source-only validation rule MRR / own validation rule MRR, both on the
  target's f = 1.00 training graph.
- **Candidate:** among candidates that pass V1–V4, the one with the larger A. Ties go to C-rand.
- **Primary fraction:** f = 0.10, fixed. It is the most data-limited level, and Study S gains
  were largest at its scarcest levels.
- **Go** only if all of the following hold:
  - (i) V1–V4 pass.
  - (ii) A ≥ 0.9. The only family with real transfer had A = 1.01; every family without it had
    A ≤ 0.73.
  - (iii) ULTRA need, from the validation-only smoke test (one seed): local fine-tuning's best
    validation MRR at f = 0.10 is at least 0.03 below its best at f = 1.00. Otherwise a +0.03
    gain would require federation to beat complete local data.
  - (iv) The automated leakage and coverage tests pass on the written files.
- **No-go** otherwise, with no fallback, as before.

These are plausibility screens, not predictions. Mobile and ICS met (iii) but failed (ii). The
smoke test's FedAvg validation numbers are reported but do not enter the decision.
