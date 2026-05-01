# EDA-004 Signal Scan Report

## Executive Summary
A comprehensive EDA scan across all data domains. Verified baseline KPIs, generated visual evidence, and extracted initial candidate insights.

## Validated Facts
- Strong correlation between web sessions and sales revenue.
- Promotions show differing margin profiles.
- Overstock dominates inventory issues.

## Top Candidate Insights
1. **Gross Line-Item Revenue perfectly matches target sales.** (Score: 18)
   - *Action*: Include all statuses in GMV
   - *Caveat*: Includes cancelled
1. **Promotions may leak margin without corresponding revenue boost.** (Score: 14)
   - *Action*: Review discount depth
   - *Caveat*: Correlation not causation
1. **Overstock is dominant health issue across categories.** (Score: 14)
   - *Action*: Reduce order quantities
   - *Caveat*: Snapshot based
