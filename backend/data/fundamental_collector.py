"""
Alpha Vantage Fundamental Data Collector.

Fetches company fundamentals (OVERVIEW, INCOME_STATEMENT, BALANCE_SHEET,
CASH_FLOW, EARNINGS) with 24-hour disk caching to stay within free-tier
API rate limits.

Used by the investment advisor to compute Buffett and Jhunjhunwala scores.
"""

import os
import json
import time
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)


class FundamentalCollector:
    """Fetches and caches fundamental financial data from Alpha Vantage."""

    BASE_URL = "https://www.alphavantage.co/query"
    CACHE_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fundamentals"
    )
    DISK_CACHE_TTL = 86400  # 24 hours

    FUNCTIONS = [
        "OVERVIEW",
        "INCOME_STATEMENT",
        "BALANCE_SHEET",
        "CASH_FLOW",
        "EARNINGS",
    ]

    def __init__(self) -> None:
        self.api_key = os.getenv("alphavantage_apikey") or os.getenv(
            "ALPHA_VANTAGE_KEY", ""
        )
        self._cache: Dict[Tuple[str, str], Tuple[float, Any]] = {}
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _safe_filename(symbol: str) -> str:
        return symbol.replace("=", "_").replace("^", "_").replace("/", "_")

    @staticmethod
    def _safe_float(value: Any, default: float = None) -> Optional[float]:
        """Convert Alpha Vantage string value to float safely."""
        if value is None or value == "None" or value == "-" or value == "":
            return default
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return default
            return v
        except (ValueError, TypeError):
            return default

    def _disk_cache_path(self, function: str, symbol: str) -> str:
        return os.path.join(
            self.CACHE_DIR,
            f"{self._safe_filename(symbol)}_{function}.json",
        )

    def _get_cached(self, function: str, symbol: str) -> Optional[Dict]:
        """Check in-memory then disk cache."""
        key = (function, symbol.upper())

        # In-memory check
        entry = self._cache.get(key)
        if entry:
            cached_at, data = entry
            if (time.time() - cached_at) <= self.DISK_CACHE_TTL:
                return data

        # Disk cache check
        path = self._disk_cache_path(function, symbol)
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                if (time.time() - mtime) <= self.DISK_CACHE_TTL:
                    with open(path, "r") as f:
                        data = json.load(f)
                    self._cache[key] = (time.time(), data)
                    return data
            except Exception:
                pass

        return None

    def _set_cached(self, function: str, symbol: str, data: Dict) -> None:
        """Store in memory and on disk."""
        key = (function, symbol.upper())
        self._cache[key] = (time.time(), data)
        path = self._disk_cache_path(function, symbol)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write cache {path}: {e}")

    def _fetch_av(self, function: str, symbol: str) -> Optional[Dict]:
        """Make a single Alpha Vantage API call."""
        if not self.api_key:
            return None
        try:
            response = requests.get(
                self.BASE_URL,
                params={
                    "function": function,
                    "symbol": symbol.upper(),
                    "apikey": self.api_key,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()

            if "Error Message" in payload:
                logger.warning(
                    f"AV {function} error for {symbol}: {payload['Error Message']}"
                )
                return None
            if "Note" in payload:
                logger.warning(f"AV rate limit hit for {symbol}: {payload['Note']}")
                return None
            if "Information" in payload and "premium" in payload.get("Information", "").lower():
                logger.warning(f"AV premium endpoint for {function}: {payload['Information']}")
                return None

            return payload
        except Exception as exc:
            logger.warning(f"AV {function} fetch failed for {symbol}: {exc}")
            return None

    # ── public endpoints ──────────────────────────────────────────

    def get_overview(self, symbol: str) -> Optional[Dict]:
        """Fetch company overview (~59 fields: PE, ROE, margins, etc.)."""
        cached = self._get_cached("OVERVIEW", symbol)
        if cached is not None:
            return cached

        data = self._fetch_av("OVERVIEW", symbol)
        if data and data.get("Symbol"):
            self._set_cached("OVERVIEW", symbol, data)
            logger.info(f"Fetched OVERVIEW for {symbol}")
            return data
        return None

    def get_income_statement(self, symbol: str) -> Optional[Dict]:
        """Fetch income statement (5yr annual + quarterly reports)."""
        cached = self._get_cached("INCOME_STATEMENT", symbol)
        if cached is not None:
            return cached

        data = self._fetch_av("INCOME_STATEMENT", symbol)
        if data and (data.get("annualReports") or data.get("quarterlyReports")):
            self._set_cached("INCOME_STATEMENT", symbol, data)
            logger.info(f"Fetched INCOME_STATEMENT for {symbol}")
            return data
        return None

    def get_balance_sheet(self, symbol: str) -> Optional[Dict]:
        """Fetch balance sheet (assets, liabilities, equity)."""
        cached = self._get_cached("BALANCE_SHEET", symbol)
        if cached is not None:
            return cached

        data = self._fetch_av("BALANCE_SHEET", symbol)
        if data and (data.get("annualReports") or data.get("quarterlyReports")):
            self._set_cached("BALANCE_SHEET", symbol, data)
            logger.info(f"Fetched BALANCE_SHEET for {symbol}")
            return data
        return None

    def get_cash_flow(self, symbol: str) -> Optional[Dict]:
        """Fetch cash flow statement (operating CF, CapEx, dividends)."""
        cached = self._get_cached("CASH_FLOW", symbol)
        if cached is not None:
            return cached

        data = self._fetch_av("CASH_FLOW", symbol)
        if data and (data.get("annualReports") or data.get("quarterlyReports")):
            self._set_cached("CASH_FLOW", symbol, data)
            logger.info(f"Fetched CASH_FLOW for {symbol}")
            return data
        return None

    def get_earnings(self, symbol: str) -> Optional[Dict]:
        """Fetch earnings data (EPS actual vs estimates)."""
        cached = self._get_cached("EARNINGS", symbol)
        if cached is not None:
            return cached

        data = self._fetch_av("EARNINGS", symbol)
        if data and (data.get("annualEarnings") or data.get("quarterlyEarnings")):
            self._set_cached("EARNINGS", symbol, data)
            logger.info(f"Fetched EARNINGS for {symbol}")
            return data
        return None

    def get_all_fundamentals(
        self, symbol: str, rate_limit_delay: float = 12.5
    ) -> Optional[Dict]:
        """Fetch all 5 fundamental endpoints, skipping cached ones.

        Returns combined dict with keys: overview, income_statement,
        balance_sheet, cash_flow, earnings.
        """
        fetchers = {
            "overview": self.get_overview,
            "income_statement": self.get_income_statement,
            "balance_sheet": self.get_balance_sheet,
            "cash_flow": self.get_cash_flow,
            "earnings": self.get_earnings,
        }

        result = {}
        api_calls = 0

        for key, func in fetchers.items():
            # Check if cached (func will return from cache if available)
            fn_name = key.upper()
            cached = self._get_cached(fn_name, symbol)
            needs_api = cached is None

            if needs_api and api_calls > 0:
                time.sleep(rate_limit_delay)

            data = func(symbol)
            result[key] = data

            if needs_api and data is not None:
                api_calls += 1

        has_any = any(v is not None for v in result.values())
        if not has_any:
            logger.warning(f"No fundamental data available for {symbol}")
            return None

        logger.info(
            f"Fundamentals for {symbol}: "
            f"{sum(1 for v in result.values() if v)}/5 endpoints "
            f"({api_calls} API calls)"
        )
        return result

    # ── derived metrics ──────────────────────────────────────────

    @staticmethod
    def _cagr(start: Optional[float], end: Optional[float], years: int) -> Optional[float]:
        """Compound annual growth rate."""
        if not start or not end or years <= 0 or start <= 0:
            return None
        try:
            return (end / start) ** (1.0 / years) - 1
        except (ZeroDivisionError, ValueError):
            return None

    def compute_derived_metrics(self, fundamentals: Dict) -> Dict[str, Optional[float]]:
        """Compute ~35 derived financial metrics from raw fundamental data."""
        sf = self._safe_float
        m: Dict[str, Optional[float]] = {}

        overview = fundamentals.get("overview") or {}
        income = fundamentals.get("income_statement") or {}
        balance = fundamentals.get("balance_sheet") or {}
        cashflow = fundamentals.get("cash_flow") or {}
        earnings = fundamentals.get("earnings") or {}

        # ── Valuation (from OVERVIEW) ─────────────────────────────
        m["pe_ratio"] = sf(overview.get("PERatio"))
        m["forward_pe"] = sf(overview.get("ForwardPE"))
        m["pb_ratio"] = sf(overview.get("PriceToBookRatio"))
        m["peg_ratio"] = sf(overview.get("PEGRatio"))
        m["ev_to_ebitda"] = sf(overview.get("EVToEBITDA"))
        m["price_to_sales"] = sf(overview.get("PriceToSalesRatioTTM"))
        m["ev_to_revenue"] = sf(overview.get("EVToRevenue"))
        m["market_cap"] = sf(overview.get("MarketCapitalization"))
        m["beta"] = sf(overview.get("Beta"))
        m["analyst_target"] = sf(overview.get("AnalystTargetPrice"))
        m["book_value"] = sf(overview.get("BookValue"))
        m["eps"] = sf(overview.get("EPS"))

        # ── Profitability (from OVERVIEW) ─────────────────────────
        m["roe_ttm"] = sf(overview.get("ReturnOnEquityTTM"))
        m["roa_ttm"] = sf(overview.get("ReturnOnAssetsTTM"))
        m["profit_margin"] = sf(overview.get("ProfitMargin"))
        m["operating_margin"] = sf(overview.get("OperatingMarginTTM"))

        # ── Dividends (from OVERVIEW) ─────────────────────────────
        m["dividend_yield"] = sf(overview.get("DividendYield"))
        m["dividend_per_share"] = sf(overview.get("DividendPerShare"))

        # ── Income Statement Metrics ──────────────────────────────
        annual_reports = income.get("annualReports", [])

        if annual_reports:
            latest = annual_reports[0]
            revenue = sf(latest.get("totalRevenue"))
            gross_profit = sf(latest.get("grossProfit"))
            op_income = sf(latest.get("operatingIncome"))
            net_income = sf(latest.get("netIncome"))
            interest_exp = sf(latest.get("interestExpense"))
            ebitda = sf(latest.get("ebitda"))

            # Gross margin
            if revenue and gross_profit and revenue > 0:
                m["gross_margin"] = gross_profit / revenue
            else:
                m["gross_margin"] = None

            # Interest coverage
            if op_income and interest_exp and interest_exp > 0:
                m["interest_coverage"] = op_income / interest_exp
            else:
                m["interest_coverage"] = None

            # Revenue CAGR
            if len(annual_reports) >= 4:
                rev_old = sf(annual_reports[3].get("totalRevenue"))
                m["revenue_cagr_3y"] = self._cagr(rev_old, revenue, 3)
            else:
                m["revenue_cagr_3y"] = None

            if len(annual_reports) >= 5:
                rev_old = sf(annual_reports[4].get("totalRevenue"))
                m["revenue_cagr_5y"] = self._cagr(rev_old, revenue, min(4, len(annual_reports) - 1))
            else:
                m["revenue_cagr_5y"] = m.get("revenue_cagr_3y")

            # EPS CAGR (from income statement net income)
            if len(annual_reports) >= 4:
                ni_old = sf(annual_reports[3].get("netIncome"))
                if ni_old and net_income and ni_old > 0 and net_income > 0:
                    m["eps_cagr_3y"] = self._cagr(ni_old, net_income, 3)
                else:
                    m["eps_cagr_3y"] = None
            else:
                m["eps_cagr_3y"] = None

            if len(annual_reports) >= 5:
                ni_old = sf(annual_reports[4].get("netIncome"))
                if ni_old and net_income and ni_old > 0 and net_income > 0:
                    m["eps_cagr_5y"] = self._cagr(ni_old, net_income, min(4, len(annual_reports) - 1))
                else:
                    m["eps_cagr_5y"] = None
            else:
                m["eps_cagr_5y"] = m.get("eps_cagr_3y")

            # YoY earnings growth
            if len(annual_reports) >= 2:
                ni_prev = sf(annual_reports[1].get("netIncome"))
                if ni_prev and net_income and ni_prev != 0:
                    m["earnings_growth_yoy"] = (net_income - ni_prev) / abs(ni_prev)
                else:
                    m["earnings_growth_yoy"] = None
            else:
                m["earnings_growth_yoy"] = None

            # Margin expansion (3y)
            if len(annual_reports) >= 4:
                pm_old = sf(annual_reports[3].get("netIncome"))
                rev_old = sf(annual_reports[3].get("totalRevenue"))
                if pm_old and rev_old and rev_old > 0 and revenue and revenue > 0 and net_income:
                    margin_old = pm_old / rev_old
                    margin_new = net_income / revenue
                    m["margin_expansion_3y"] = margin_new - margin_old
                else:
                    m["margin_expansion_3y"] = None
            else:
                m["margin_expansion_3y"] = None

            # Operating margin trend (slope over available years)
            op_margins = []
            for r in reversed(annual_reports[:5]):
                rev_r = sf(r.get("totalRevenue"))
                op_r = sf(r.get("operatingIncome"))
                if rev_r and op_r and rev_r > 0:
                    op_margins.append(op_r / rev_r)
            if len(op_margins) >= 3:
                x = np.arange(len(op_margins))
                slope = np.polyfit(x, op_margins, 1)[0]
                m["operating_margin_trend"] = float(slope)
            else:
                m["operating_margin_trend"] = None
        else:
            for k in [
                "gross_margin", "interest_coverage", "revenue_cagr_3y",
                "revenue_cagr_5y", "eps_cagr_3y", "eps_cagr_5y",
                "earnings_growth_yoy", "margin_expansion_3y",
                "operating_margin_trend",
            ]:
                m[k] = None

        # ── Balance Sheet Metrics ─────────────────────────────────
        bs_reports = balance.get("annualReports", [])

        if bs_reports:
            bs_latest = bs_reports[0]
            total_assets = sf(bs_latest.get("totalAssets"))
            total_liabilities = sf(bs_latest.get("totalLiabilities"))
            total_equity = sf(bs_latest.get("totalShareholderEquity"))
            current_assets = sf(bs_latest.get("totalCurrentAssets"))
            current_liabilities = sf(bs_latest.get("totalCurrentLiabilities"))
            long_term_debt = sf(bs_latest.get("longTermDebt"), 0)
            current_debt = sf(bs_latest.get("currentDebt"), 0)
            short_term_debt = sf(bs_latest.get("shortTermDebt"), 0)
            cash = sf(bs_latest.get("cashAndCashEquivalentsAtCarryingValue"), 0)

            total_debt = (long_term_debt or 0) + (current_debt or 0) + (short_term_debt or 0)

            # Debt-to-equity
            if total_equity and total_equity > 0:
                m["debt_to_equity"] = total_debt / total_equity
            else:
                m["debt_to_equity"] = None

            # Current ratio
            if current_liabilities and current_liabilities > 0 and current_assets:
                m["current_ratio"] = current_assets / current_liabilities
            else:
                m["current_ratio"] = None

            # Net debt to EBITDA
            ebitda_val = sf((income.get("annualReports") or [{}])[0].get("ebitda")) if income.get("annualReports") else None
            net_debt = total_debt - (cash or 0)
            if ebitda_val and ebitda_val > 0:
                m["net_debt_to_ebitda"] = net_debt / ebitda_val
            else:
                m["net_debt_to_ebitda"] = None

            # ROIC = net income / (equity + long_term_debt - cash)
            net_inc = sf((income.get("annualReports") or [{}])[0].get("netIncome")) if income.get("annualReports") else None
            invested = (total_equity or 0) + (long_term_debt or 0) - (cash or 0)
            if net_inc and invested > 0:
                m["roic"] = net_inc / invested
            else:
                m["roic"] = None

            # ROE from annual balance sheets (5y avg, std, trend)
            roe_vals = []
            for i, bs_r in enumerate(bs_reports[:5]):
                eq = sf(bs_r.get("totalShareholderEquity"))
                ni_r = sf((income.get("annualReports") or [{}] * (i + 1))[min(i, len(income.get("annualReports", [])) - 1)].get("netIncome")) if income.get("annualReports") and i < len(income.get("annualReports", [])) else None
                if eq and ni_r and eq > 0:
                    roe_vals.append(ni_r / eq)

            if roe_vals:
                m["roe_5y_avg"] = float(np.mean(roe_vals))
                m["roe_5y_std"] = float(np.std(roe_vals)) if len(roe_vals) > 1 else 0.0
                if len(roe_vals) >= 3:
                    x = np.arange(len(roe_vals))
                    m["roe_trend"] = float(np.polyfit(x, list(reversed(roe_vals)), 1)[0])
                else:
                    m["roe_trend"] = None
            else:
                m["roe_5y_avg"] = m.get("roe_ttm")
                m["roe_5y_std"] = None
                m["roe_trend"] = None
        else:
            for k in [
                "debt_to_equity", "current_ratio", "net_debt_to_ebitda",
                "roic", "roe_5y_avg", "roe_5y_std", "roe_trend",
            ]:
                m[k] = None

        # ── Cash Flow Metrics ─────────────────────────────────────
        cf_reports = cashflow.get("annualReports", [])

        if cf_reports:
            cf_latest = cf_reports[0]
            op_cf = sf(cf_latest.get("operatingCashflow"))
            capex = sf(cf_latest.get("capitalExpenditures"))
            div_paid = sf(cf_latest.get("dividendPayout"))
            buyback = sf(cf_latest.get("paymentsForRepurchaseOfCommonStock"))
            net_inc_cf = sf(cf_latest.get("netIncome"))
            revenue_cf = sf((income.get("annualReports") or [{}])[0].get("totalRevenue")) if income.get("annualReports") else None

            # CapEx is often reported as positive but represents cash outflow
            capex_abs = abs(capex) if capex else 0

            # Free cash flow
            if op_cf is not None:
                m["fcf"] = op_cf - capex_abs
            else:
                m["fcf"] = None

            # FCF yield
            mkt_cap = m.get("market_cap")
            if m["fcf"] and mkt_cap and mkt_cap > 0:
                m["fcf_yield"] = m["fcf"] / mkt_cap
            else:
                m["fcf_yield"] = None

            # FCF-to-net-income conversion
            if net_inc_cf and net_inc_cf > 0 and m["fcf"]:
                m["fcf_to_net_income"] = m["fcf"] / net_inc_cf
            else:
                m["fcf_to_net_income"] = None

            # CapEx to revenue
            if revenue_cf and revenue_cf > 0:
                m["capex_to_revenue"] = capex_abs / revenue_cf
            else:
                m["capex_to_revenue"] = None

            # Buyback yield
            if buyback and mkt_cap and mkt_cap > 0:
                m["buyback_yield"] = abs(buyback) / mkt_cap
            else:
                m["buyback_yield"] = 0.0

            # Total shareholder yield
            m["total_yield"] = (m.get("dividend_yield") or 0) + (m.get("buyback_yield") or 0)

            # Dividend payout ratio
            if div_paid and net_inc_cf and net_inc_cf > 0:
                m["dividend_payout_ratio"] = abs(div_paid) / net_inc_cf
            else:
                m["dividend_payout_ratio"] = None
        else:
            for k in [
                "fcf", "fcf_yield", "fcf_to_net_income", "capex_to_revenue",
                "buyback_yield", "total_yield", "dividend_payout_ratio",
            ]:
                m[k] = None

        # ── Earnings Quality ──────────────────────────────────────
        q_earnings = earnings.get("quarterlyEarnings", [])
        a_earnings = earnings.get("annualEarnings", [])

        # EPS consistency (inverse of coefficient of variation over annual EPS)
        if a_earnings and len(a_earnings) >= 3:
            eps_vals = [sf(e.get("reportedEPS")) for e in a_earnings[:5] if sf(e.get("reportedEPS")) is not None]
            if eps_vals and len(eps_vals) >= 3:
                mean_eps = np.mean(eps_vals)
                std_eps = np.std(eps_vals)
                if mean_eps > 0:
                    cv = std_eps / mean_eps
                    m["earnings_consistency"] = max(0, min(1, 1 - cv))
                else:
                    m["earnings_consistency"] = 0
            else:
                m["earnings_consistency"] = None
        else:
            m["earnings_consistency"] = None

        # Earnings beat rate (last 8 quarters)
        if q_earnings:
            recent = q_earnings[:8]
            beats = 0
            total = 0
            surprise_sum = 0
            for q in recent:
                surprise_pct = sf(q.get("surprisePercentage"))
                if surprise_pct is not None:
                    total += 1
                    surprise_sum += surprise_pct
                    if surprise_pct > 0:
                        beats += 1
            m["earnings_beat_rate"] = beats / total if total > 0 else None
            m["earnings_surprise_avg"] = surprise_sum / total if total > 0 else None
        else:
            m["earnings_beat_rate"] = None
            m["earnings_surprise_avg"] = None

        return m
