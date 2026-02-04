import math
import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Should We Contract This? (TN)", layout="wide")

# =============================
# CONFIG
# =============================
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1XVrJ1lz-oIf9AjnKtyuzb8PmH9zk2nPy1aZ1VEotubA/edit?gid=0#gid=0"

# -----------------------------
# Helpers
# -----------------------------
def normalize_status(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    if s == "sold":
        return "sold"
    if s in {"cut loose", "cutloose", "cut"}:
        return "cut"
    return s

def effective_price_row(row) -> float:
    amended = row.get("Amended Price", None)
    contract = row.get("Contract Price", None)
    if pd.notna(amended):
        return float(amended)
    if pd.notna(contract):
        return float(contract)
    return float("nan")

def dollars(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"${x:,.0f}"

def confidence_label(total_n: int) -> str:
    if total_n >= 30:
        return "✅ High"
    if total_n >= 15:
        return "⚠️ Medium"
    return "🚧 Low"

def auto_params_for_county(total_n: int) -> tuple[int, int, int]:
    """
    Returns (step, tail_min_n, min_bin_n) based on sample size.
    step: increment used to scan thresholds (decision logic)
    tail_min_n: minimum number of deals in the tail (>= threshold) to trust cliff
    min_bin_n: minimum deals per bin for the context table
    """
    if total_n >= 120:
        return (5000, 20, 5)
    if total_n >= 60:
        return (10000, 15, 4)
    if total_n >= 30:
        return (10000, 10, 3)
    if total_n >= 15:
        return (15000, 8, 3)
    return (20000, 6, 2)

def google_sheet_to_csv_url(url: str) -> str:
    """
    Convert a standard Google Sheets URL to a CSV export URL.
    Works with URLs like:
      https://docs.google.com/spreadsheets/d/<ID>/edit?gid=<GID>#gid=<GID>
    """
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Could not find spreadsheet ID in the URL.")
    sheet_id = m.group(1)

    gid_match = re.search(r"[?&#]gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"

    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

@st.cache_data(ttl=300)  # cache for 5 minutes
def load_data_from_google_sheet(sheet_url: str) -> pd.DataFrame:
    csv_url = google_sheet_to_csv_url(sheet_url)
    # If sharing is not enabled, this will error (403/404)
    return pd.read_csv(csv_url)

def build_bins(df_county: pd.DataFrame, bin_size: int, min_bin_n: int) -> pd.DataFrame:
    """
    Context table only (NOT used for decision thresholds).
    Returns: bin_low, bin_high, n, cut_rate
    """
    prices = pd.to_numeric(df_county["effective_price"], errors="coerce").dropna()
    if prices.empty:
        return pd.DataFrame(columns=["bin_low", "bin_high", "n", "cut_rate"])

    pmin, pmax = float(prices.min()), float(prices.max())
    start = int(math.floor(pmin / bin_size) * bin_size)
    end = int(math.ceil(pmax / bin_size) * bin_size)

    bins = list(range(start, end + bin_size, bin_size))
    if len(bins) < 3:
        bins = [start, end + bin_size]

    df = df_county.copy()
    df["effective_price"] = pd.to_numeric(df["effective_price"], errors="coerce")
    df = df.dropna(subset=["effective_price"])

    df["price_bin"] = pd.cut(df["effective_price"], bins=bins, right=True, include_lowest=True)

    grp = (
        df.groupby("price_bin", observed=False)
        .agg(
            n=("status_norm", "size"),
            cut_rate=("is_cut", "mean"),
        )
        .reset_index()
    )

    grp["bin_low"] = grp["price_bin"].apply(lambda x: float(x.left) if pd.notna(x) else float("nan"))
    grp["bin_high"] = grp["price_bin"].apply(lambda x: float(x.right) if pd.notna(x) else float("nan"))

    grp["bin_low"] = pd.to_numeric(grp["bin_low"], errors="coerce")
    grp["bin_high"] = pd.to_numeric(grp["bin_high"], errors="coerce")
    grp["cut_rate"] = pd.to_numeric(grp["cut_rate"], errors="coerce")
    grp = grp.dropna(subset=["bin_low", "bin_high", "cut_rate"])

    grp = grp[grp["n"] >= min_bin_n].copy()
    grp = grp.sort_values(["bin_low"]).reset_index(drop=True)

    return grp[["bin_low", "bin_high", "n", "cut_rate"]]

def find_tail_threshold(df_county: pd.DataFrame, target_cut_rate: float, tail_min_n: int, step: int) -> float | None:
    """
    Find LOWEST price P such that among deals with effective_price >= P,
    cut rate >= target_cut_rate AND count >= tail_min_n.
    """
    d = df_county.copy()
    d["effective_price"] = pd.to_numeric(d["effective_price"], errors="coerce")
    d = d.dropna(subset=["effective_price", "is_cut"])
    if d.empty:
        return None

    prices = d["effective_price"].astype(float)
    pmin, pmax = float(prices.min()), float(prices.max())

    start = int((pmin // step) * step)
    end = int(((pmax + step - 1) // step) * step)

    for P in range(start, end + step, step):
        tail = d[d["effective_price"] >= P]
        n = len(tail)
        if n < tail_min_n:
            continue
        cut_rate = float(tail["is_cut"].mean())
        if cut_rate >= target_cut_rate:
            return float(P)

    return None

# -----------------------------
# UI
# -----------------------------
st.title("✅ Should We Contract This? — TN (simple)")

st.markdown(
    """
This app pulls directly from your Google Sheet and outputs **Green / Yellow / Red**.

- It uses **effective contract price** = (Amended Price if present, else Contract Price) *behind the scenes*.
- Acquisitions only enters: **Proposed Contract Price**.
"""
)

# Quick refresh button
col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("🔄 Refresh data"):
        load_data_from_google_sheet.clear()  # clears cache

# Load sheet
try:
    df_raw = load_data_from_google_sheet(GOOGLE_SHEET_URL)
except Exception as e:
    st.error(
        "Couldn't load the Google Sheet. Most common cause: the sheet isn't shared as "
        "**Anyone with the link can view**.\n\n"
        f"Details: {e}"
    )
    st.stop()

# Validate columns
required_cols = ["County", "Status", "Contract Price", "Amended Price", "Market"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Your sheet is missing these required columns: {missing}")
    st.stop()

# Prep dataframe
df = df_raw.copy()
df["status_norm"] = df["Status"].apply(normalize_status)
df = df[df["status_norm"].isin(["sold", "cut"])].copy()

df["effective_price"] = df.apply(effective_price_row, axis=1)
df["effective_price"] = pd.to_numeric(df["effective_price"], errors="coerce")
df = df.dropna(subset=["effective_price"])

df["is_cut"] = (df["status_norm"] == "cut").astype(int)
df["is_sold"] = (df["status_norm"] == "sold").astype(int)

# Sidebar controls (simple)
st.sidebar.header("Inputs")

market_options = sorted(df["Market"].dropna().unique().tolist())
default_market_idx = market_options.index("Nashville/Middle TN") if "Nashville/Middle TN" in market_options else 0
market = st.sidebar.selectbox("Market", market_options, index=default_market_idx)

df_m = df[df["Market"] == market].copy()

county_options = sorted(df_m["County"].dropna().unique().tolist())
county = st.sidebar.selectbox("County", county_options)

contract_price = st.sidebar.number_input("Proposed Contract Price ($)", min_value=0, value=150000, step=5000)
input_price = float(contract_price)

# Filter to county
cdf = df_m[df_m["County"] == county].copy()

total_n = len(cdf)
sold_n = int(cdf["is_sold"].sum())
cut_n = int(cdf["is_cut"].sum())

# Auto tuning behind the scenes
step, tail_min_n, min_bin_n = auto_params_for_county(total_n)

# County stats
avg_sold = cdf.loc[cdf["is_sold"] == 1, "effective_price"].mean()

# Context table (auto)
bin_stats = build_bins(cdf, bin_size=step, min_bin_n=min_bin_n)

# Decision thresholds (auto)
line_80 = find_tail_threshold(cdf, 0.80, tail_min_n=tail_min_n, step=step)
line_90 = find_tail_threshold(cdf, 0.90, tail_min_n=tail_min_n, step=step)

conf = confidence_label(total_n)

# Build "Why"
reason = []
if not math.isnan(avg_sold):
    reason.append(f"Avg SOLD effective price: {dollars(avg_sold)}")

if line_80 is not None:
    t80 = cdf[cdf["effective_price"] >= line_80]
    reason.append(f"~80% cut cliff around: {dollars(line_80)} (Deals ≥ line: {len(t80)}, cut rate: {(t80['is_cut'].mean()*100):.0f}%)")

if line_90 is not None:
    t90 = cdf[cdf["effective_price"] >= line_90]
    reason.append(f"~90% cut cliff around: {dollars(line_90)} (Deals ≥ line: {len(t90)}, cut rate: {(t90['is_cut'].mean()*100):.0f}%)")

# Recommendation logic
if line_90 is not None and input_price >= line_90:
    rec = "🔴 RED — Likely Cut Loose"
elif line_80 is not None and input_price >= line_80:
    rec = "🟡 YELLOW — Caution / Needs justification"
else:
    # fallback only when we can't compute cliffs
    if not math.isnan(avg_sold) and input_price <= avg_sold * 1.10:
        rec = "🟢 GREEN — Contractable"
    elif not math.isnan(avg_sold) and input_price >= avg_sold * 1.35:
        rec = "🔴 RED — Likely Cut Loose"
    else:
        rec = "🟡 YELLOW — Caution / Needs justification"

# Layout
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Decision")
    st.markdown(f"### {rec}")
    st.write(f"**Input contract price:** {dollars(input_price)}")
    st.write(f"**County sample:** {total_n} deals  |  **Sold:** {sold_n}  |  **Cut Loose:** {cut_n}")
    st.write(f"**Confidence:** {conf}")

    if conf == "🚧 Low":
        st.warning("Low data volume in this county. Use as guidance only; get buyer alignment to confirm pricing.")

    st.markdown("**Why:**")
    if reason:
        for r in reason:
            st.write(f"- {r}")
    else:
        st.write("- Not enough data to compute thresholds.")

    if line_90 is not None and input_price >= line_90:
        st.error("This is in the **90%+ cut zone** for this county. Strongly avoid unless the deal is exceptional.")
    elif line_80 is not None and input_price >= line_80:
        st.warning("This is in the **80% cut zone**. Only sign with clear justification.")
    else:
        st.success("This price is *not* in the high-failure zone based on your historical outcomes.")

with right:
    st.subheader("Cut-Rate by Price Range (context)")
    if bin_stats.empty:
        st.info("Not enough data to build a context table for this county.")
    else:
        show = bin_stats.copy()
        show["Price Range"] = show.apply(lambda r: f"{dollars(r['bin_low'])}–{dollars(r['bin_high'])}", axis=1)
        show["Cut Rate"] = (show["cut_rate"] * 100).round(0).astype(int).astype(str) + "%"
        show = show[["Price Range", "n", "Cut Rate"]].rename(columns={"n": "Deals in bin"})
        st.dataframe(show, use_container_width=True)

st.divider()
st.caption("Data refreshes automatically every ~5 minutes (or click Refresh).")
