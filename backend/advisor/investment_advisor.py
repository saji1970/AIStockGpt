"""
Investment Philosophy Scoring -- Warren Buffett & Rakesh Jhunjhunwala.

Scores stocks 0-100 against the investment criteria of two legendary
investors, using fundamental data from Alpha Vantage.

BuffettScore: Value investing, economic moats, margin of safety.
JhunjhunwalaScore: Growth at reasonable price (GARP), turnaround, multibagger.
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Grade helpers ─────────────────────────────────────────────────

def _grade(score: int) -> str:
    if score >= 90: return "A+"
    if score >= 85: return "A"
    if score >= 80: return "A-"
    if score >= 75: return "B+"
    if score >= 70: return "B"
    if score >= 65: return "B-"
    if score >= 60: return "C+"
    if score >= 55: return "C"
    if score >= 50: return "C-"
    if score >= 40: return "D"
    return "F"


def _clamp(value: float, lo: float = 0, hi: float = 100) -> int:
    return int(min(hi, max(lo, value)))


def _safe(val: Optional[float], default: float = 0) -> float:
    return val if val is not None else default


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1%}"


def _fmt_ratio(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


# ══════════════════════════════════════════════════════════════════
#  WARREN BUFFETT SCORING
# ══════════════════════════════════════════════════════════════════

class BuffettScore:
    """Score a stock 0-100 on Warren Buffett's investment criteria.

    Sub-criteria (weights):
        Moat                25 %
        Value               25 %
        Financial Health    20 %
        Earnings Quality    15 %
        Management          15 %
    """

    WEIGHTS = {
        "moat": 0.25,
        "value": 0.25,
        "financial_health": 0.20,
        "earnings_quality": 0.15,
        "management": 0.15,
    }

    VERDICTS = [
        (85, "Exceptional Buffett Pick"),
        (75, "Strong Buffett Pick"),
        (65, "Moderate Buffett Alignment"),
        (55, "Weak Buffett Alignment"),
        (0, "Not a Buffett-Style Investment"),
    ]

    def __init__(self, fundamental_collector, av_collector=None):
        self.fc = fundamental_collector
        self.av = av_collector

    def score(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Compute Buffett score for *symbol*. Returns None if no data."""
        fundamentals = self.fc.get_all_fundamentals(symbol)
        if fundamentals is None:
            return None

        metrics = self.fc.compute_derived_metrics(fundamentals)

        moat, moat_d, moat_n = self._score_moat(metrics)
        value, value_d, value_n = self._score_value(metrics)
        health, health_d, health_n = self._score_financial_health(metrics)
        quality, quality_d, quality_n = self._score_earnings_quality(metrics)
        mgmt, mgmt_d, mgmt_n = self._score_management(metrics)

        overall = _clamp(
            moat * self.WEIGHTS["moat"]
            + value * self.WEIGHTS["value"]
            + health * self.WEIGHTS["financial_health"]
            + quality * self.WEIGHTS["earnings_quality"]
            + mgmt * self.WEIGHTS["management"]
        )

        verdict = next(v for t, v in self.VERDICTS if overall >= t)

        narrative = (
            f"{symbol} scores {overall}/100 on the Buffett scale ({_grade(overall)}). "
            f"{verdict}. "
            f"Moat strength is {'strong' if moat >= 70 else 'moderate' if moat >= 50 else 'weak'} "
            f"({moat}/100). "
            f"Valuation is {'attractive' if value >= 70 else 'fair' if value >= 50 else 'stretched'} "
            f"({value}/100). "
            f"Financial health is {'excellent' if health >= 80 else 'solid' if health >= 60 else 'concerning'} "
            f"({health}/100)."
        )

        return {
            "symbol": symbol,
            "overall_score": overall,
            "grade": _grade(overall),
            "verdict": verdict,
            "moat_score": moat,
            "moat_details": moat_d,
            "moat_narrative": moat_n,
            "value_score": value,
            "value_details": value_d,
            "value_narrative": value_n,
            "financial_health_score": health,
            "financial_health_details": health_d,
            "financial_health_narrative": health_n,
            "earnings_quality_score": quality,
            "earnings_quality_details": quality_d,
            "earnings_quality_narrative": quality_n,
            "management_score": mgmt,
            "management_details": mgmt_d,
            "management_narrative": mgmt_n,
            "narrative": narrative,
            "metrics": metrics,
        }

    # ── sub-scores ────────────────────────────────────────────────

    def _score_moat(self, m: Dict) -> Tuple[int, Dict, str]:
        """Economic moat: durable competitive advantage."""
        roe_avg = _safe(m.get("roe_5y_avg"))
        roe_std = _safe(m.get("roe_5y_std"), 0.5)
        pm = _safe(m.get("profit_margin"))
        rev_cagr = _safe(m.get("revenue_cagr_5y"))

        # ROE consistency: high average + low variability
        if roe_avg > 0:
            stability = max(0, 1 - min(roe_std / max(roe_avg, 0.01), 1))
            roe_score = _clamp((roe_avg / 0.20) * 50 + stability * 50)
        else:
            roe_score = 0

        # Margin stability
        margin_score = _clamp(pm / 0.20 * 100)

        # Revenue durability
        rev_score = _clamp(50 + rev_cagr / 0.15 * 50)

        moat = _clamp(roe_score * 0.45 + margin_score * 0.30 + rev_score * 0.25)

        details = {
            "roe_consistency": roe_score,
            "margin_stability": margin_score,
            "revenue_durability": rev_score,
        }
        narrative = (
            f"ROE {_fmt_pct(m.get('roe_5y_avg'))} avg (std {_fmt_pct(m.get('roe_5y_std'))}), "
            f"profit margin {_fmt_pct(m.get('profit_margin'))}, "
            f"revenue CAGR {_fmt_pct(m.get('revenue_cagr_5y'))}"
        )
        return moat, details, narrative

    def _score_value(self, m: Dict) -> Tuple[int, Dict, str]:
        """Value: buying below intrinsic value."""
        peg = _safe(m.get("peg_ratio"), 3)
        pb = _safe(m.get("pb_ratio"), 5)
        fcf_yield = _safe(m.get("fcf_yield"))
        ev_ebitda = _safe(m.get("ev_to_ebitda"), 20)

        peg_score = _clamp((2.5 - peg) / 2.5 * 100)
        pb_score = _clamp((5 - pb) / 4 * 100)
        fcf_score = _clamp(fcf_yield / 0.08 * 100)
        ev_score = _clamp((15 - ev_ebitda) / 15 * 100)

        value = _clamp(peg_score * 0.30 + pb_score * 0.20 + fcf_score * 0.30 + ev_score * 0.20)

        details = {
            "pe_vs_growth": peg_score,
            "pb_attractiveness": pb_score,
            "fcf_yield_value": fcf_score,
            "ev_ebitda_value": ev_score,
        }
        narrative = (
            f"PEG {_fmt_ratio(m.get('peg_ratio'))}, "
            f"P/B {_fmt_ratio(m.get('pb_ratio'))}, "
            f"FCF yield {_fmt_pct(m.get('fcf_yield'))}, "
            f"EV/EBITDA {_fmt_ratio(m.get('ev_to_ebitda'))}"
        )
        return value, details, narrative

    def _score_financial_health(self, m: Dict) -> Tuple[int, Dict, str]:
        """Financial health: low debt, strong cash generation."""
        de = _safe(m.get("debt_to_equity"), 2)
        ic = _safe(m.get("interest_coverage"))
        fcf_y = _safe(m.get("fcf_yield"))
        cr = _safe(m.get("current_ratio"))

        debt_score = _clamp((2 - de) / 2 * 100)
        ic_score = _clamp(ic / 10 * 100)
        cash_score = _clamp(fcf_y / 0.06 * 100)
        cr_score = _clamp(cr / 2 * 100)

        health = _clamp(debt_score * 0.35 + ic_score * 0.25 + cash_score * 0.25 + cr_score * 0.15)

        details = {
            "debt_safety": debt_score,
            "interest_coverage": ic_score,
            "cash_generation": cash_score,
            "current_ratio": cr_score,
        }
        narrative = (
            f"D/E {_fmt_ratio(m.get('debt_to_equity'))}, "
            f"interest coverage {_fmt_ratio(m.get('interest_coverage'))}x, "
            f"current ratio {_fmt_ratio(m.get('current_ratio'))}"
        )
        return health, details, narrative

    def _score_earnings_quality(self, m: Dict) -> Tuple[int, Dict, str]:
        """Earnings quality: consistency, beats, FCF conversion."""
        consistency = _safe(m.get("earnings_consistency"))
        beat_rate = _safe(m.get("earnings_beat_rate"), 0.5)
        fcf_conv = _safe(m.get("fcf_to_net_income"))

        cons_score = _clamp(consistency * 100)
        beat_score = _clamp(beat_rate * 100)
        conv_score = _clamp(fcf_conv / 1.2 * 100)

        quality = _clamp(cons_score * 0.30 + beat_score * 0.30 + conv_score * 0.40)

        details = {
            "consistency": cons_score,
            "beat_rate": beat_score,
            "fcf_conversion": conv_score,
        }
        narrative = (
            f"Earnings consistency {_fmt_pct(m.get('earnings_consistency'))}, "
            f"beat rate {_fmt_pct(m.get('earnings_beat_rate'))}, "
            f"FCF/NI {_fmt_ratio(m.get('fcf_to_net_income'))}"
        )
        return quality, details, narrative

    def _score_management(self, m: Dict) -> Tuple[int, Dict, str]:
        """Management: ROE trend, capital allocation, ROIC."""
        roe_trend = _safe(m.get("roe_trend"))
        total_yield = _safe(m.get("total_yield"))
        roic = _safe(m.get("roic"))

        roe_score = _clamp(50 + roe_trend / 0.05 * 50)
        alloc_score = _clamp(total_yield / 0.05 * 100)
        roic_score = _clamp(roic / 0.20 * 100)

        mgmt = _clamp(roe_score * 0.30 + alloc_score * 0.35 + roic_score * 0.35)

        details = {
            "roe_trend": roe_score,
            "capital_allocation": alloc_score,
            "roic": roic_score,
        }
        narrative = (
            f"ROE trend {'improving' if roe_trend and roe_trend > 0 else 'declining'}, "
            f"total yield {_fmt_pct(m.get('total_yield'))}, "
            f"ROIC {_fmt_pct(m.get('roic'))}"
        )
        return mgmt, details, narrative


# ══════════════════════════════════════════════════════════════════
#  RAKESH JHUNJHUNWALA SCORING
# ══════════════════════════════════════════════════════════════════

class JhunjhunwalaScore:
    """Score a stock 0-100 on Rakesh Jhunjhunwala's GARP criteria.

    Sub-criteria (weights):
        Growth              30 %
        Value-for-Growth    20 %
        Turnaround          20 %
        Multibagger         20 %
        Financial Flex      10 %
    """

    WEIGHTS = {
        "growth": 0.30,
        "value_for_growth": 0.20,
        "turnaround": 0.20,
        "multibagger": 0.20,
        "financial_flexibility": 0.10,
    }

    VERDICTS = [
        (85, "Exceptional GARP Pick"),
        (75, "Strong Jhunjhunwala-Style Pick"),
        (65, "Moderate GARP Potential"),
        (55, "Weak GARP Signal"),
        (0, "Not a GARP-Style Investment"),
    ]

    def __init__(self, fundamental_collector, av_collector=None):
        self.fc = fundamental_collector
        self.av = av_collector

    def score(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Compute Jhunjhunwala score for *symbol*. Returns None if no data."""
        fundamentals = self.fc.get_all_fundamentals(symbol)
        if fundamentals is None:
            return None

        metrics = self.fc.compute_derived_metrics(fundamentals)

        growth, growth_d, growth_n = self._score_growth(metrics)
        vfg, vfg_d, vfg_n = self._score_value_for_growth(metrics)
        turn, turn_d, turn_n = self._score_turnaround(metrics)
        multi, multi_d, multi_n = self._score_multibagger(metrics)
        flex, flex_d, flex_n = self._score_financial_flexibility(metrics)

        overall = _clamp(
            growth * self.WEIGHTS["growth"]
            + vfg * self.WEIGHTS["value_for_growth"]
            + turn * self.WEIGHTS["turnaround"]
            + multi * self.WEIGHTS["multibagger"]
            + flex * self.WEIGHTS["financial_flexibility"]
        )

        verdict = next(v for t, v in self.VERDICTS if overall >= t)

        narrative = (
            f"{symbol} scores {overall}/100 on the Jhunjhunwala GARP scale ({_grade(overall)}). "
            f"{verdict}. "
            f"Growth is {'exceptional' if growth >= 80 else 'strong' if growth >= 60 else 'moderate' if growth >= 40 else 'weak'} "
            f"({growth}/100). "
            f"Turnaround potential is {'high' if turn >= 70 else 'moderate' if turn >= 50 else 'low'} "
            f"({turn}/100). "
            f"Multibagger potential is {'high' if multi >= 70 else 'moderate' if multi >= 50 else 'limited'} "
            f"({multi}/100)."
        )

        return {
            "symbol": symbol,
            "overall_score": overall,
            "grade": _grade(overall),
            "verdict": verdict,
            "growth_score": growth,
            "growth_details": growth_d,
            "growth_narrative": growth_n,
            "value_for_growth_score": vfg,
            "value_for_growth_details": vfg_d,
            "value_for_growth_narrative": vfg_n,
            "turnaround_score": turn,
            "turnaround_details": turn_d,
            "turnaround_narrative": turn_n,
            "multibagger_potential": multi,
            "multibagger_details": multi_d,
            "multibagger_narrative": multi_n,
            "financial_flexibility_score": flex,
            "financial_flexibility_details": flex_d,
            "financial_flexibility_narrative": flex_n,
            "narrative": narrative,
            "metrics": metrics,
        }

    # ── sub-scores ────────────────────────────────────────────────

    def _score_growth(self, m: Dict) -> Tuple[int, Dict, str]:
        """Growth: revenue, earnings, and margin expansion."""
        rev_cagr = _safe(m.get("revenue_cagr_3y"))
        eps_cagr = _safe(m.get("eps_cagr_3y"))
        margin_exp = _safe(m.get("margin_expansion_3y"))

        rev_score = _clamp(rev_cagr / 0.25 * 100)
        eps_score = _clamp(eps_cagr / 0.25 * 100)
        margin_score = _clamp(50 + margin_exp / 0.05 * 50)

        growth = _clamp(rev_score * 0.35 + eps_score * 0.40 + margin_score * 0.25)

        details = {
            "revenue_growth": rev_score,
            "earnings_growth": eps_score,
            "margin_expansion": margin_score,
        }
        narrative = (
            f"Revenue CAGR {_fmt_pct(m.get('revenue_cagr_3y'))}, "
            f"earnings CAGR {_fmt_pct(m.get('eps_cagr_3y'))}, "
            f"margin expansion {_fmt_pct(m.get('margin_expansion_3y'))}"
        )
        return growth, details, narrative

    def _score_value_for_growth(self, m: Dict) -> Tuple[int, Dict, str]:
        """GARP: growth at reasonable price."""
        peg = _safe(m.get("peg_ratio"), 3)
        fwd_pe = _safe(m.get("forward_pe"), 30)
        eps_cagr = _safe(m.get("eps_cagr_3y"), 0.10)

        peg_score = _clamp((2 - peg) / 2 * 100)

        # Forward PE relative to growth rate
        if eps_cagr > 0:
            pe_to_growth = fwd_pe / (eps_cagr * 100)
            ptg_score = _clamp((2 - pe_to_growth) / 2 * 100)
        else:
            ptg_score = 0

        vfg = _clamp(peg_score * 0.50 + ptg_score * 0.50)

        details = {
            "peg_score": peg_score,
            "pe_to_growth": ptg_score,
        }
        narrative = (
            f"PEG {_fmt_ratio(m.get('peg_ratio'))}, "
            f"forward PE {_fmt_ratio(m.get('forward_pe'))}"
        )
        return vfg, details, narrative

    def _score_turnaround(self, m: Dict) -> Tuple[int, Dict, str]:
        """Turnaround: improving trajectory."""
        margin_trend = _safe(m.get("operating_margin_trend"))
        surprise_avg = _safe(m.get("earnings_surprise_avg"))
        roe_trend = _safe(m.get("roe_trend"))

        margin_imp = _clamp(50 + margin_trend / 0.03 * 50)
        surprise_score = _clamp(50 + surprise_avg / 10 * 50)
        roe_imp = _clamp(50 + roe_trend / 0.03 * 50)

        turn = _clamp(margin_imp * 0.40 + surprise_score * 0.30 + roe_imp * 0.30)

        details = {
            "margin_improvement": margin_imp,
            "earnings_surprise": surprise_score,
            "roe_improvement": roe_imp,
        }
        narrative = (
            f"Op. margin trend {'positive' if margin_trend and margin_trend > 0 else 'negative'}, "
            f"avg earnings surprise {_fmt_pct(m.get('earnings_surprise_avg')) if m.get('earnings_surprise_avg') else 'N/A'}%, "
            f"ROE trend {'improving' if roe_trend and roe_trend > 0 else 'declining'}"
        )
        return turn, details, narrative

    def _score_multibagger(self, m: Dict) -> Tuple[int, Dict, str]:
        """Multibagger potential: acceleration, leverage, scale."""
        rev_3y = _safe(m.get("revenue_cagr_3y"))
        rev_5y = _safe(m.get("revenue_cagr_5y"))
        op_trend = _safe(m.get("operating_margin_trend"))

        # Revenue acceleration
        accel = rev_3y - rev_5y
        accel_score = _clamp(50 + accel / 0.10 * 50)

        # Operating leverage
        leverage_score = _clamp(50 + op_trend / 0.03 * 50)

        # Absolute growth above 20%
        growth_above_20 = _clamp(rev_3y / 0.20 * 100)

        multi = _clamp(accel_score * 0.30 + leverage_score * 0.30 + growth_above_20 * 0.40)

        details = {
            "revenue_acceleration": accel_score,
            "operating_leverage": leverage_score,
            "high_growth": growth_above_20,
        }
        narrative = (
            f"Revenue 3Y CAGR {_fmt_pct(m.get('revenue_cagr_3y'))} "
            f"vs 5Y {_fmt_pct(m.get('revenue_cagr_5y'))}, "
            f"operating margin trend {'expanding' if op_trend and op_trend > 0 else 'contracting'}"
        )
        return multi, details, narrative

    def _score_financial_flexibility(self, m: Dict) -> Tuple[int, Dict, str]:
        """Financial flexibility: low debt + cash generation for growth."""
        de = _safe(m.get("debt_to_equity"), 2)
        fcf_yield = _safe(m.get("fcf_yield"))

        debt_score = _clamp((1.5 - de) / 1.5 * 100)
        cash_score = _clamp(fcf_yield / 0.05 * 100)

        flex = _clamp(debt_score * 0.50 + cash_score * 0.50)

        details = {
            "low_debt": debt_score,
            "cash_generation": cash_score,
        }
        narrative = (
            f"D/E {_fmt_ratio(m.get('debt_to_equity'))}, "
            f"FCF yield {_fmt_pct(m.get('fcf_yield'))}"
        )
        return flex, details, narrative


# ══════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ══════════════════════════════════════════════════════════════════

def compare_philosophies(
    symbol: str,
    fundamental_collector,
    av_collector=None,
) -> Optional[Dict[str, Any]]:
    """Run both Buffett and Jhunjhunwala scoring for a single stock."""
    buffett = BuffettScore(fundamental_collector, av_collector).score(symbol)
    rj = JhunjhunwalaScore(fundamental_collector, av_collector).score(symbol)

    if buffett is None and rj is None:
        return None

    b_score = buffett["overall_score"] if buffett else 0
    r_score = rj["overall_score"] if rj else 0

    return {
        "symbol": symbol,
        "buffett": buffett,
        "jhunjhunwala": rj,
        "better_fit": "buffett" if b_score >= r_score else "jhunjhunwala",
    }
