"""
Natural Gas Storage Contract Pricing Model (Prototype)
=======================================================
Prices a gas storage contract: buy (inject) gas on a set of dates, hold it in
storage, then sell (withdraw) it on a later set of dates, capturing the
seasonal price differential (e.g., buy in summer when prices are low, sell in
winter when prices are high).

Contract value = (revenue from all withdrawals)
                - (cost of all injections)
                - (injection/withdrawal handling costs)
                - (storage/rental costs)
                - (transport costs)

Assumptions (per task spec):
  - No transport delay -- gas bought on a date is available in storage
    immediately, and gas sold on a date leaves storage immediately.
  - Interest rates are zero (no discounting of cash flows).
  - Weekends, market holidays, and bank holidays are not considered --
    any calendar date is a valid trading date.
  - Injection/withdrawal is bounded by a maximum daily rate and the facility
    has a hard capacity limit that can never be exceeded.

The pricer works generally: any number of injection dates and withdrawal
dates, in any order, each with its own volume and price. It walks through
all events in chronological order, tracks the running inventory, validates
that rate and capacity constraints are respected, and accumulates all cash
flows.

This file is self-contained and can be run/submitted on its own. It
optionally uses `nat_gas_pricing.py` (from the earlier task) in Test 2 to
pull seasonally-estimated prices, but falls back to hardcoded sample prices
if that module isn't present, so this script has no hard dependency.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd


@dataclass
class StorageEvent:
    date: pd.Timestamp
    action: str          # "inject" or "withdraw"
    volume: float         # MMBtu moved on this date (positive number)
    price: float           # $/MMBtu on this date


@dataclass
class ContractResult:
    value: float
    cash_flows: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Cash flow breakdown:"]
        for cf in self.cash_flows:
            lines.append(
                f"  {cf['date'].date()}  {cf['type']:<22} {cf['amount']:>14,.2f}"
            )
        lines.append(f"{'':32}{'-'*20}")
        lines.append(f"{'Net contract value':32}{self.value:>14,.2f}")
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def price_storage_contract(
    injection_dates: List[Union[str, datetime]],
    withdrawal_dates: List[Union[str, datetime]],
    injection_prices: List[float],
    withdrawal_prices: List[float],
    injection_volumes: List[float],
    withdrawal_volumes: List[float],
    max_injection_withdrawal_rate: float,
    max_storage_volume: float,
    storage_cost_per_month: float,
    injection_withdrawal_cost_rate: float,
    transport_cost_per_event: float = 0.0,
) -> ContractResult:
    """
    Price a natural gas storage contract with arbitrary injection/withdrawal
    schedules.

    Parameters
    ----------
    injection_dates : list of dates gas is purchased and put into storage.
    withdrawal_dates : list of dates gas is taken out of storage and sold.
    injection_prices : $/MMBtu purchase price on each injection date
                        (same length & order as injection_dates).
    withdrawal_prices : $/MMBtu sale price on each withdrawal date
                        (same length & order as withdrawal_dates).
    injection_volumes : MMBtu injected on each injection date.
    withdrawal_volumes : MMBtu withdrawn on each withdrawal date.
    max_injection_withdrawal_rate : maximum MMBtu that can be injected OR
                        withdrawn in a single event/day.
    max_storage_volume : maximum MMBtu the facility can hold at any time.
    storage_cost_per_month : fixed $ rental fee charged per calendar month
                        (or part thereof) that the facility is used, from
                        first injection to last withdrawal.
    injection_withdrawal_cost_rate : $ per MMBtu charged by the facility
                        each time gas is injected AND each time gas is
                        withdrawn (i.e., applied on both legs).
    transport_cost_per_event : fixed $ cost incurred each time gas is
                        physically transported to/from the facility (charged
                        once per injection event and once per withdrawal
                        event).

    Returns
    -------
    ContractResult
        `.value`      -- net value of the contract in $ (revenue - costs)
        `.cash_flows` -- itemized list of every cash flow, for transparency
        `.warnings`   -- any constraint violations detected (e.g., attempting
                         to inject beyond capacity, or withdraw more than is
                         in storage) -- these events are skipped/clipped
                         rather than silently allowed.
    """
    n_in, n_out = len(injection_dates), len(withdrawal_dates)
    assert n_in == len(injection_prices) == len(injection_volumes), \
        "injection_dates, injection_prices, injection_volumes must be same length"
    assert n_out == len(withdrawal_prices) == len(withdrawal_volumes), \
        "withdrawal_dates, withdrawal_prices, withdrawal_volumes must be same length"

    events = []
    for d, p, v in zip(injection_dates, injection_prices, injection_volumes):
        events.append(StorageEvent(pd.Timestamp(d), "inject", v, p))
    for d, p, v in zip(withdrawal_dates, withdrawal_prices, withdrawal_volumes):
        events.append(StorageEvent(pd.Timestamp(d), "withdraw", v, p))

    # Process strictly in chronological order; if same date, inject before
    # withdraw (arbitrary but consistent tie-break).
    events.sort(key=lambda e: (e.date, e.action == "withdraw"))

    volume = 0.0
    cash_flows = []
    warnings = []

    for e in events:
        if e.volume > max_injection_withdrawal_rate:
            warnings.append(
                f"{e.date.date()}: requested {e.action} of {e.volume:,.0f} MMBtu "
                f"exceeds max rate of {max_injection_withdrawal_rate:,.0f} "
                f"MMBtu/event -- volume clipped to max rate."
            )
            e.volume = max_injection_withdrawal_rate

        if e.action == "inject":
            available_capacity = max_storage_volume - volume
            if e.volume > available_capacity:
                warnings.append(
                    f"{e.date.date()}: requested injection of {e.volume:,.0f} "
                    f"MMBtu exceeds available capacity of "
                    f"{available_capacity:,.0f} MMBtu -- volume clipped."
                )
                e.volume = available_capacity

            if e.volume <= 0:
                continue

            volume += e.volume
            purchase_cost = e.volume * e.price
            handling_cost = e.volume * injection_withdrawal_cost_rate

            cash_flows.append({"date": e.date, "type": "Gas purchase (injection)", "amount": -purchase_cost})
            cash_flows.append({"date": e.date, "type": "Injection handling fee", "amount": -handling_cost})
            if transport_cost_per_event:
                cash_flows.append({"date": e.date, "type": "Transport cost (injection)", "amount": -transport_cost_per_event})

        else:  # withdraw
            if e.volume > volume:
                warnings.append(
                    f"{e.date.date()}: requested withdrawal of {e.volume:,.0f} "
                    f"MMBtu exceeds available stored volume of {volume:,.0f} "
                    f"MMBtu -- volume clipped."
                )
                e.volume = volume

            if e.volume <= 0:
                continue

            volume -= e.volume
            sale_revenue = e.volume * e.price
            handling_cost = e.volume * injection_withdrawal_cost_rate

            cash_flows.append({"date": e.date, "type": "Gas sale (withdrawal)", "amount": sale_revenue})
            cash_flows.append({"date": e.date, "type": "Withdrawal handling fee", "amount": -handling_cost})
            if transport_cost_per_event:
                cash_flows.append({"date": e.date, "type": "Transport cost (withdrawal)", "amount": -transport_cost_per_event})

    # Storage/rental cost: charged for every calendar month (or part thereof)
    # between the first injection and the last withdrawal.
    if events:
        first_date = min(e.date for e in events if e.action == "inject") if injection_dates else min(e.date for e in events)
        last_date = max(e.date for e in events if e.action == "withdraw") if withdrawal_dates else max(e.date for e in events)
        months_stored = (last_date.year - first_date.year) * 12 + (last_date.month - first_date.month)
        # Round any partial month up to a full month of rent (standard rental convention).
        if last_date.day > first_date.day or months_stored == 0:
            months_stored += 1
        months_stored = max(months_stored, 1)
        storage_cost = months_stored * storage_cost_per_month
        cash_flows.append({
            "date": first_date,
            "type": f"Storage rental ({months_stored} month(s))",
            "amount": -storage_cost,
        })

    net_value = sum(cf["amount"] for cf in cash_flows)
    cash_flows.sort(key=lambda cf: cf["date"])

    return ContractResult(value=net_value, cash_flows=cash_flows, warnings=warnings)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Test 1: Reproduce the worked example from the task description
    #   Buy 1,000,000 MMBtu in summer at $2/MMBtu, store 4 months,
    #   sell at $3/MMBtu, $100K/month rent, $10K/1M MMBtu inject+withdraw,
    #   $50K/event transport.
    #   Expected: $1,000,000 (spread) - $400,000 (rent) - $20,000 (handling)
    #             - $100,000 (transport) = $480,000
    # ------------------------------------------------------------------
    result1 = price_storage_contract(
        injection_dates=["2024-06-01"],
        withdrawal_dates=["2024-10-01"],
        injection_prices=[2.0],
        withdrawal_prices=[3.0],
        injection_volumes=[1_000_000],
        withdrawal_volumes=[1_000_000],
        max_injection_withdrawal_rate=1_000_000,
        max_storage_volume=1_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_rate=0.01,   # $10K / 1,000,000 MMBtu = $0.01/MMBtu
        transport_cost_per_event=50_000,
    )
    print("=== Test 1: Single buy/sell cycle (matches worked example) ===")
    print(result1.summary())
    print()

    # ------------------------------------------------------------------
    # Test 2: Multiple injection/withdrawal dates. Client wants to buy gas
    # now (thinking winter will be colder than normal) across several
    # months and sell it down across the winter. Prices come from the
    # seasonal price estimator built in the earlier task if available;
    # otherwise we fall back to representative sample prices so this
    # script runs standalone.
    # ------------------------------------------------------------------
    inj_dates = ["2024-06-30", "2024-07-31", "2024-08-31"]
    wd_dates = ["2024-12-31", "2025-01-31", "2025-02-28"]

    try:
        from nat_gas_pricing import NatGasPriceEstimator
        est = NatGasPriceEstimator("Nat_Gas.csv")
        inj_prices = [est.estimate(d) for d in inj_dates]
        wd_prices = [est.estimate(d) for d in wd_dates]
    except Exception:
        # Fallback sample prices (representative of the seasonal pattern)
        # used only if nat_gas_pricing.py / Nat_Gas.csv aren't available.
        inj_prices = [11.50, 11.60, 11.50]
        wd_prices = [13.00, 13.15, 13.12]

    result2 = price_storage_contract(
        injection_dates=inj_dates,
        withdrawal_dates=wd_dates,
        injection_prices=inj_prices,
        withdrawal_prices=wd_prices,
        injection_volumes=[500_000, 500_000, 500_000],
        withdrawal_volumes=[500_000, 500_000, 500_000],
        max_injection_withdrawal_rate=500_000,
        max_storage_volume=1_500_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_rate=0.01,
        transport_cost_per_event=50_000,
    )
    print("=== Test 2: Multi-date injection/withdrawal using estimated prices ===")
    print("Injection prices:", [f"${p:.2f}" for p in inj_prices])
    print("Withdrawal prices:", [f"${p:.2f}" for p in wd_prices])
    print(result2.summary())
    print()

    # ------------------------------------------------------------------
    # Test 3: Constraint violation -- try to inject more than capacity
    # allows, to show the model flags and clips rather than silently
    # over-filling the facility.
    # ------------------------------------------------------------------
    result3 = price_storage_contract(
        injection_dates=["2024-06-01", "2024-07-01"],
        withdrawal_dates=["2024-12-01"],
        injection_prices=[2.0, 2.1],
        withdrawal_prices=[3.2],
        injection_volumes=[800_000, 800_000],
        withdrawal_volumes=[1_600_000],
        max_injection_withdrawal_rate=800_000,
        max_storage_volume=1_000_000,   # capacity smaller than total requested injections
        storage_cost_per_month=50_000,
        injection_withdrawal_cost_rate=0.01,
        transport_cost_per_event=25_000,
    )
    print("=== Test 3: Capacity-constrained scenario (should show warnings) ===")
    print(result3.summary())
