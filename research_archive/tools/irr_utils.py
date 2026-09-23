# tools/irr_utils.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import re, zipfile

# Minimal xirr
def _xnpv(rate: float, dates, amounts) -> float:
    dates = list(dates); amounts = list(amounts)
    if not dates or not amounts or len(dates) != len(amounts):
        return np.nan
    if rate <= -0.999999:   # prevents complex base when (1+rate) <= 0
        return np.inf
    t0 = dates[0]
    acc = 0.0
    for a, d in zip(amounts, dates):
        days = (d - t0).days / 365.25
        acc += a / ((1.0 + rate) ** days)
    return acc

def xirr(dates, amounts, guess: float = 0.15, maxiter: int = 100, tol: float = 1e-6) -> float:
    dates = list(dates); amounts = list(amounts)
    if len(dates) < 2 or not any(a > 0 for a in amounts) or not any(a < 0 for a in amounts):
        return np.nan
    # bounds
    lo, hi = -0.9999, 10.0

    # try Newton within bounds
    r = min(max(guess, lo + 1e-6), hi - 1e-6)
    for _ in range(maxiter):
        f  = _xnpv(r, dates, amounts)
        f1 = _xnpv(r + 1e-6, dates, amounts)
        if not np.isfinite(f) or not np.isfinite(f1):
            break
        dfdx = (f1 - f) / 1e-6
        if dfdx == 0:
            break
        r_next = r - f / dfdx
        # keep inside bounds
        r_next = min(max(r_next, lo + 1e-6), hi - 1e-6)
        if abs(r_next - r) < tol:
            return r_next
        r = r_next

    # robust bracketed bisection
    f_lo = _xnpv(lo + 1e-6, dates, amounts)
    f_hi = _xnpv(hi - 1e-6, dates, amounts)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return np.nan
    # If no sign change, try expanding hi a bit
    if f_lo * f_hi > 0:
        # attempt scan on upper bound
        for new_hi in [20.0, 50.0]:
            f_hi = _xnpv(new_hi - 1e-6, dates, amounts)
            if np.isfinite(f_hi) and f_lo * f_hi <= 0:
                hi = new_hi
                break
        else:
            return np.nan

    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = _xnpv(mid, dates, amounts)
        if not np.isfinite(f_mid):
            mid = max(mid, lo + 1e-6)
            f_mid = _xnpv(mid, dates, amounts)
            if not np.isfinite(f_mid):
                return np.nan
        if abs(hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0

def _assert_valid_xlsx(path: Path):
    if not path or not str(path): return
    if not path.exists(): raise FileNotFoundError(path)
    if path.suffix.lower()==".xlsx":
        try:
            with zipfile.ZipFile(path) as z:
                if "[Content_Types].xml" not in z.namelist():
                    raise ValueError(f"Invalid .xlsx: {path}")
        except zipfile.BadZipFile:
            raise ValueError(f"Unreadable .xlsx: {path}")

def load_universe_if_any(root: Path, cfg: dict) -> pd.DataFrame | None:
    uf = cfg["irr"].get("universe_file","").strip()
    if not uf:
        return None
    path = root / uf
    _assert_valid_xlsx(path)
    sheet = cfg["irr"].get("universe_sheet") or 0
    xls = pd.ExcelFile(path, engine="openpyxl")
    use_sheet = sheet if (isinstance(sheet,str) and sheet in xls.sheet_names) else sheet
    df = pd.read_excel(xls, sheet_name=use_sheet, engine="openpyxl")
    # normalize expected names
    ren = {
        "Deal ID":"DealID","Companies":"Company","Deal Date":"DealDate","Deal Type":"DealType",
        "Deal Type 2":"DealType2","Deal Type 3":"DealType3","Deal Size":"DealSize",
        "Deal Synopsis":"DealSynopsis","Deal Status":"DealStatus"
    }
    df = df.rename(columns={c:ren[c] for c in df.columns if c in ren})
    if "DealDate" in df.columns:
        df["DealDate"] = pd.to_datetime(df["DealDate"], errors="coerce")
    if "Company" in df.columns:
        df["Company_norm"] = (
            df["Company"].astype(str).str.lower().str.strip()
              .str.replace(r"[\,\.\-]"," ", regex=True).str.replace(r"\s+"," ", regex=True)
        )
    return df

def mark_distributions(events: pd.DataFrame, cfg_irr: dict) -> pd.DataFrame:
    if events is None:
        return None
    df = events.copy()

    labels = [s.lower() for s in cfg_irr.get("recap_labels", [])]
    syn_re  = cfg_irr.get("recap_regex_synopsis") or None

    lab = pd.Series(False, index=df.index)

    for col in ("DealType", "DealType2", "DealType3"):
        if col in df.columns:
            lab = lab | df[col].astype(str).str.lower().isin(labels)

    if syn_re:
        # convert any (...) to non-capturing (?:...) to avoid pandas warning
        safe_pat = re.sub(r"\((?!\?)", r"(?:", syn_re)
        syn_col = next((c for c in ["DealSynopsis", "Deal Synopsis", "Synopsis"] if c in df.columns), None)
        if syn_col:
            lab = lab | df[syn_col].astype(str).str.contains(safe_pat, regex=True, na=False)

    df["IsDistributionEvent"] = lab
    amt_col = cfg_irr.get("distribution_amount_col", "DealSize")
    df["DistributionAmount"] = pd.to_numeric(df.get(amt_col, np.nan), errors="coerce")
    return df

def deal_cashflows_from_pairs(pairs: pd.DataFrame, universe_marked: pd.DataFrame | None, cfg_irr: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cashflows_long, per_deal_metrics). If universe_marked is None, distributions = 0."""
    dist_scale = float(cfg_irr.get("distribution_scale",1.0))
    min_days   = int(cfg_irr.get("min_hold_days",0))
    max_years  = cfg_irr.get("max_hold_years", None)

    recs, metrics = [], []
    pairs_iter = pairs.reset_index(drop=True).copy()
    # EXPECT: columns Entry_DealDate, Exit_DealDate, Entry_DealSize, Exit_DealSize already present
    for i, row in pairs_iter.iterrows():
        # respect completeness rule
        if not (pd.notna(row.get("Entry_DealDate")) and pd.notna(row.get("Exit_DealDate"))
                and pd.notna(row.get("Entry_DealSize")) and pd.notna(row.get("Exit_DealSize"))):
            # still return a metrics row with NaN IRR / MOIC
            metrics.append({
                "PairKey": i,
                "Company": row.get("Entry_Company"),
                "EntryDate": row.get("Entry_DealDate"),
                "ExitDate": row.get("Exit_DealDate"),
                "Entry_DealSize": row.get("Entry_DealSize"),
                "Exit_DealSize": row.get("Exit_DealSize"),
                "TotalDistributions": np.nan,
                "TotalGains": np.nan,
                "IRR_XIRR": np.nan,
                "MOIC": np.nan,
                "HoldingYears": ((pd.to_datetime(row.get("Exit_DealDate")) - pd.to_datetime(
                    row.get("Entry_DealDate"))).days / 365.25) if (
                            pd.notna(row.get("Entry_DealDate")) and pd.notna(row.get("Exit_DealDate"))) else np.nan,
            })
            continue

        company = row.get("Entry_Company")
        comp_norm = str(company).lower().strip() if company is not None else ""
        ed = pd.to_datetime(row["Entry_DealDate"])
        xd = pd.to_datetime(row["Exit_DealDate"])
        e_amt = float(row["Entry_DealSize"])
        x_amt = float(row["Exit_DealSize"])

        dates = [ed]
        amts = [-e_amt]
        recs.append({"PairKey": i, "Company": company, "Date": ed, "Amount": -e_amt, "Kind": "Entry"})

        if universe_marked is not None:
            mid = universe_marked[
                (universe_marked.get("Company_norm", "") == comp_norm) &
                (universe_marked["DealDate"] > ed) &
                (universe_marked["DealDate"] < xd) &
                (universe_marked["IsDistributionEvent"])
                ][["DealDate", "DistributionAmount"]].dropna()
            scale = float(cfg_irr.get("distribution_scale", 1.0))
            for _, r2 in mid.sort_values("DealDate").iterrows():
                amt = float(r2["DistributionAmount"]) * scale
                dates.append(r2["DealDate"])
                amts.append(amt)
                recs.append(
                    {"PairKey": i, "Company": company, "Date": r2["DealDate"], "Amount": amt, "Kind": "Distribution"})

        dates.append(xd)
        amts.append(x_amt)
        recs.append({"PairKey": i, "Company": company, "Date": xd, "Amount": x_amt, "Kind": "Exit"})

        irr = xirr(dates, amts)
        inflows = sum(a for a in amts if a > 0)
        total_dist = inflows - x_amt
        moic = (inflows / e_amt) if e_amt else np.nan

        metrics.append({
            "PairKey": i,
            "Company": company,
            "EntryDate": ed,
            "ExitDate": xd,
            "Entry_DealSize": e_amt,
            "Exit_DealSize": x_amt,
            "TotalDistributions": total_dist,
            "TotalGains": total_dist + x_amt,
            "IRR_XIRR": irr,
            "MOIC": moic,
            "HoldingYears": (xd - ed).days / 365.25,
        })

    return pd.DataFrame.from_records(recs), pd.DataFrame.from_records(metrics)

    # if universe available, ensure Company_norm
    if universe_marked is not None and "Company_norm" not in universe_marked.columns and "Company" in universe_marked.columns:
        universe_marked["Company_norm"] = universe_marked["Company"].astype(str).str.lower().str.strip()

    for i, row in pairs.reset_index(drop=True).iterrows():
        company = row["Entry_Company"]
        comp_norm = str(company).lower().strip()
        ed = pd.to_datetime(row["Entry_DealDate"])
        xd = pd.to_datetime(row["Exit_DealDate"])
        e_amt = float(row.get("Entry_DealSize") or 0.0)
        x_amt = float(row.get("Exit_DealSize") or 0.0)
        if pd.isna(ed) or pd.isna(xd):
            continue
        hold_days = (xd - ed).days
        if min_days and hold_days < min_days:
            continue
        if max_years is not None and (hold_days/365.25) > max_years:
            continue

        dates = [ed]; amts = [-e_amt]
        recs.append({"PairKey":i,"Company":company,"Date":ed,"Amount":-e_amt,"Kind":"Entry"})

        if universe_marked is not None:
            mid = universe_marked[
                (universe_marked.get("Company_norm","")==comp_norm) &
                (universe_marked["DealDate"] > ed) &
                (universe_marked["DealDate"] < xd) &
                (universe_marked["IsDistributionEvent"])
            ][["DealDate","DistributionAmount"]].dropna()
            for _, r2 in mid.sort_values("DealDate").iterrows():
                amt = float(r2["DistributionAmount"])*dist_scale
                dates.append(r2["DealDate"]); amts.append(amt)
                recs.append({"PairKey":i,"Company":company,"Date":r2["DealDate"],"Amount":amt,"Kind":"Distribution"})

        dates.append(xd); amts.append(x_amt)
        recs.append({"PairKey":i,"Company":company,"Date":xd,"Amount":x_amt,"Kind":"Exit"})

        irr = xirr(dates, amts)
        inflows = sum(a for a in amts if a>0)
        total_dist = inflows - x_amt
        outflow = -min(0.0, min(amts)) if any(a<0 for a in amts) else e_amt
        moic = (inflows / outflow) if outflow else np.nan

        metrics.append({
            "PairKey": i,
            "Company": company,
            "EntryDate": ed,
            "ExitDate": xd,
            "Entry_DealSize": e_amt,
            "Exit_DealSize": x_amt,
            "TotalDistributions": total_dist,
            "TotalGains": total_dist + x_amt,
            "IRR_XIRR": irr,
            "MOIC": moic,
            "HoldingYears": hold_days/365.25,
        })

    return pd.DataFrame.from_records(recs), pd.DataFrame.from_records(metrics)
