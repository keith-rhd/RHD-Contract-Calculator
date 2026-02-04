import math
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Should We Contract This? (TN)", layout="wide")

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

def load_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    # Expect these columns from your sheet:
    # Address, City, County, Status, Contract Price, Amended Price, Market, Date (optional)
    return df

def effective_price_row(row) -> float:
    amended = row.get("Amended Price", None)
    contract = row.get("Contract Price", None)
    if pd.notna(amended):
        return float(amended)
    if pd.notna(contract):
        return float(contract)
    return float("nan")

def build_bins(df_county: pd.DataFrame, bin_size: int, min_bin_n: int) -> pd.DataFrame:
    """
    Build $-increment bins and compute cut rate per bin.
    Returns a dataframe with: bin_low, bin_high, n, cut_rate
    """
    prices = df_county["effective_price"].dropna()
    if prices.empty:
        return pd.DataFrame(columns=["bin_low", "bin_high", "n", "cut_rate"])

    pmin, pmax = prices.min(), prices.max()
    # Round to bin edges
    start = int(math.floor(pmin / bin_size) * bin_size)
    end = int(math.ceil(pmax / bin_size) * bin_size)

    bins = list(range(start, end + bin_size, bin_size))
    if len(bins) < 3:
        # too little spread; just one bin
        bins = [start, end + bin_size]

    df = df_county.copy()
    df["price_bin"] = pd.cut(df["effective_price"], bins=bins, right=True, include_lowest=True)

    grp = (
        df.groupby("price_bin", observed=False)
        .agg(
            n=("status_norm", "size"),
            cut_rate=("is_cut", "mean"),
        )
        .reset_index()
    )

    # Pull numeric edges from Interval
    grp["bin_low"] = grp["price_bin"].apply(lambda x: float(x.left) if pd.notna(x) else float("nan"))
    grp["bin_high"] = grp["price_bin"].apply(lambda x: float(x.right) if pd.notna(x) else float("nan"))

    grp = grp[grp["n"] >= min_bin_n].copy()
    grp = grp.sort_values(["bin_low"]).reset_index(drop=True)
    grp["bin_low"] = pd.to_numeric(grp["bin_low"], errors="coerce")
    grp["bin_high"] = pd.to_numeric(grp["bin_high"], errors="coerce")
    return grp[["bin_low", "bin_high", "n", "cut_rate"]]

def find_threshold(bin_stats: pd.DataFrame, target_cut_rate: float) -> float | None:
    """
    Find the LOWEST bin_low where cut_rate >= target_cut_rate.
    Return that bin_low as the 'line' (threshold). None if not found.
    """
    if bin_stats.empty:
        return None
    hits = bin_stats[bin_stats["cut_rate"] >= target_cut_rate]
    if hits.empty:
        return None
    return float(hits.iloc[0]["bin_low"])

def dollars(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"${x:,.0f}"

# -----------------------------
# UI
# -----------------------------
st.title("✅ Should We Contract This? — TN (data-driven)")

st.markdown(
    """
This tool uses your historical outcomes to estimate where deals *stop selling* based on **effective contract price**
(**Amended Price if present, otherwise Contract Price**).
"""
)

uploaded = st.file_uploader("Upload your Excel file (Properties Sold In TN.xlsx)", type=["xlsx"])

if not uploaded:
    st.info("Upload the Excel file to begin.")
    st.stop()

df_raw = load_data(uploaded)

required_cols = ["County", "Status", "Contract Price", "Amended Price", "Market"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Your file is missing these required columns: {missing}")
    st.stop()

df = df_raw.copy()
df["status_norm"] = df["Status"].apply(normalize_status)
df = df[df["status_norm"].isin(["sold", "cut"])].copy()

df["effective_price"] = df.apply(effective_price_row, axis=1)
df = df[pd.notna(df["effective_price"])].copy()

df["is_cut"] = (df["status_norm"] == "cut").astype(int)
df["is_sold"] = (df["status_norm"] == "sold").astype(int)

# Sidebar controls
st.sidebar.header("Inputs")

market_options = sorted(df["Market"].dropna().unique().tolist())
market = st.sidebar.selectbox("Market", market_options, index=market_options.index("Nashville/Middle TN") if "Nashville/Middle TN" in market_options else 0)

df_m = df[df["Market"] == market].copy()

county_options = sorted(df_m["County"].dropna().unique().tolist())
county = st.sidebar.selectbox("County", county_options)

contract_price = st.sidebar.number_input("Contract Price ($)", min_value=0, value=150000, step=5000)
amended_price = st.sidebar.number_input("Amended Price ($) (optional)", min_value=0, value=0, step=5000)
use_amended = st.sidebar.checkbox("Use amended price", value=False)

input_price = amended_price if use_amended and amended_price > 0 else contract_price

st.sidebar.header("Tuning (optional)")
bin_size = st.sidebar.selectbox("Price bin size", [5000, 10000, 15000, 20000], index=1)
min_bin_n = st.sidebar.selectbox("Minimum deals per bin (stability)", [3, 5, 8, 10], index=1)

# Filter to county
cdf = df_m[df_m["County"] == county].copy()

total_n = len(cdf)
sold_n = int(cdf["is_sold"].sum())
cut_n = int(cdf["is_cut"].sum())

if total_n < 10:
    st.warning(f"Low sample size for {county} in {market}: only {total_n} deals. Use with caution.")

# County stats
avg_sold = cdf.loc[cdf["is_sold"] == 1, "effective_price"].mean()

bin_stats = build_bins(cdf, bin_size=bin_size, min_bin_n=min_bin_n)

line_80 = find_threshold(bin_stats, 0.80)
line_90 = find_threshold(bin_stats, 0.90)

# Recommendation logic
# Green: below 80-line (or below avg_sold if no line)
# Yellow: between 80 and 90 (or moderately above avg_sold)
# Red: at/above 90-line (or way above avg_sold)
rec = "UNKNOWN"
reason = []

if not math.isnan(avg_sold):
    reason.append(f"Avg SOLD effective price: {dollars(avg_sold)}")

if line_80 is not None:
    reason.append(f"~80% cut-rate starts around: {dollars(line_80)}")
if line_90 is not None:
    reason.append(f"~90% cut-rate starts around: {dollars(line_90)}")

# Determine bands (fallbacks if thresholds missing)
green_cap = line_80 if line_80 is not None else avg_sold * 1.10 if not math.isnan(avg_sold) else None
red_floor = line_90 if line_90 is not None else avg_sold * 1.35 if not math.isnan(avg_sold) else None

if green_cap is not None and input_price <= green_cap:
    rec = "🟢 GREEN — Contractable"
elif red_floor is not None and input_price >= red_floor:
    rec = "🔴 RED — Likely Cut Loose"
else:
    rec = "🟡 YELLOW — Caution / Needs justification"

# Layout
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Decision")
    st.markdown(f"### {rec}")
    st.write(f"**Input effective price:** {dollars(input_price)}")
    st.write(f"**County sample:** {total_n} deals  |  **Sold:** {sold_n}  |  **Cut Loose:** {cut_n}")

    st.markdown("**Why:**")
    for r in reason:
        st.write(f"- {r}")

    # Simple coaching line
    if line_90 is not None and input_price >= line_90:
        st.error("This is in the **90%+ cut zone** for this county. You’re very likely to waste time unless the deal is exceptional.")
    elif line_80 is not None and input_price >= line_80:
        st.warning("This is in the **80% cut zone**. Only sign if the property quality/location is clearly above-average or you have buyer alignment.")
    else:
        st.success("This price is *not* in the high-failure zone based on your historical outcomes.")

with right:
    st.subheader("County Cut-Rate by Price Range")
    if bin_stats.empty:
        st.info("Not enough data to build bins with the current settings.")
    else:
        show = bin_stats.copy()
        show["Price Range"] = show.apply(lambda r: f"{dollars(r['bin_low'])}–{dollars(r['bin_high'])}", axis=1)
        show["Cut Rate"] = (show["cut_rate"] * 100).round(0).astype(int).astype(str) + "%"
        show = show[["Price Range", "n", "Cut Rate"]].rename(columns={"n": "Deals in bin"})
        st.dataframe(show, use_container_width=True)

        # Highlight where the input falls (safe numeric comparisons)
        if not bin_stats.empty:
            bs = bin_stats.copy()
        
            # Force numeric dtypes (fixes occasional categorical/object weirdness)
            bs["bin_low"] = pd.to_numeric(bs["bin_low"], errors="coerce")
            bs["bin_high"] = pd.to_numeric(bs["bin_high"], errors="coerce")
            ip = float(input_price)
        
            bs = bs.dropna(subset=["bin_low", "bin_high"])
        
            match = bs[(bs["bin_low"] < ip) & (ip <= bs["bin_high"])]
        
            if not match.empty:
                cr = float(match.iloc[0]["cut_rate"])
                n = int(match.iloc[0]["n"])
                st.caption(
                    f"Your price falls in a bin with **{int(round(cr*100,0))}%** cut rate "
                    f"over **{n}** deals (bin size: ${bin_size:,})."
        )


st.divider()
st.caption("Tip: If a county looks noisy, increase Minimum deals per bin or bin size for more stable thresholds.")
