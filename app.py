# app.py - Streamlit app for Live Chat Analysis
import html
import logging
import os
import sys
import tempfile
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import auth

sys.path.append('.')
try:
    from Chat_Analyzer_System import DataPreprocessor, CompleteAnalysisPipeline, ResultsExporter
    ANALYSIS_AVAILABLE = True
    ANALYSIS_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - shown in the UI
    ANALYSIS_AVAILABLE = False
    ANALYSIS_IMPORT_ERROR = str(e)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("lca.app")

st.set_page_config(
    page_title="Live Chat Analysis",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_NAME = "Live Chat Analysis"
RAW_REQUIRED_COLUMNS = ['Ticket Number', 'Role', 'Sender', 'Message Date', 'Message']
COMPLAINT_REQUIRED_COLUMNS = ['No.Handphone', 'Lead Time (Solved)']
SAMPLE_RAW = "sample_data/raw_conversation_sample.xlsx"
SAMPLE_COMPLAINT = "sample_data/complaint_sample.xlsx"

# One semantic palette used by every chart and badge.
ISSUE_ORDER = ['NORMAL', 'SERIOUS', 'COMPLAINT']
ISSUE_COLORS = {'NORMAL': '#4C78A8', 'SERIOUS': '#D98A1F', 'COMPLAINT': '#C0392B'}
PERF_ORDER = ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']
PERF_COLORS = {'EXCELLENT': '#2E7D4F', 'GOOD': '#7FB069', 'FAIR': '#D98A1F', 'POOR': '#C0392B'}
REPLY_COLORS = {'First Reply': '#4C78A8', 'Final Reply': '#18212F'}

st.markdown("""
<style>
  .block-container { max-width: 1240px; padding-top: 2.25rem; padding-bottom: 3rem; }
  @media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.25rem; }
  }
  [data-testid="stMetric"] {
    border: 1px solid #DCE1E8; border-radius: 8px; padding: 12px 14px; background: #FFFFFF;
  }
  [data-testid="stMetricLabel"] p { color: #5B6573; font-size: 0.82rem; }
  /* top bar: keep brand and account on one row on phones */
  .st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; align-items: center; gap: 0.5rem; }
  .st-key-topbar [data-testid="stColumn"] { min-width: 0; width: auto; flex: 1 1 auto; }
  .st-key-topbar [data-testid="stColumn"]:last-child { flex: 0 0 auto; }
  .st-key-topbar { border-bottom: 1px solid #DCE1E8; padding-bottom: 0.6rem; margin-bottom: 0.5rem; }
  .st-key-topbar [data-testid="stMarkdownContainer"], .st-key-topbar [data-testid="stMarkdownContainer"] p { margin: 0 !important; }
  .st-key-topbar [data-testid="stElementContainer"] { margin-bottom: 0; }
  .st-key-sign_out button { min-height: 0; padding-top: 0; padding-bottom: 0; }
  /* KPI rows: two per row on phones instead of one tall stack */
  @media (max-width: 640px) {
    [class*="st-key-kpis"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
    [class*="st-key-kpis"] [data-testid="stColumn"] { flex: 1 1 calc(50% - 0.5rem); min-width: calc(50% - 0.5rem); }
    [data-testid="stMetric"] { padding: 10px 12px; }
  }
  .lca-brand { display: block; line-height: 24px; font-weight: 650; font-size: 1.05rem; color: #18212F; white-space: nowrap; }
  .lca-user { line-height: 24px; color: #5B6573; font-size: 0.85rem; text-align: right; overflow: hidden;
              text-overflow: ellipsis; white-space: nowrap; }
  @media (max-width: 640px) { .lca-user { display: none; } }
  /* sign-in and recovery screens */
  .st-key-auth_card { max-width: 380px; margin: 9vh auto 0 auto; }
  .lca-auth-title { font-size: 1.35rem; font-weight: 650; margin: 0 0 0.25rem 0; color: #18212F; }
  .lca-auth-sub { color: #5B6573; font-size: 0.92rem; margin-bottom: 1rem; }
  .lca-mark { width: 36px; height: 36px; border-radius: 8px; background: #1F5FAD; color: #fff;
              display: flex; align-items: center; justify-content: center; font-weight: 700;
              font-size: 0.95rem; margin-bottom: 1rem; letter-spacing: 0.02em; }
  .lca-quote { border-left: 3px solid #4C78A8; background: #F3F5F8; padding: 10px 12px;
               border-radius: 4px; margin: 0.25rem 0 0.75rem 0; white-space: pre-wrap; }
  .lca-badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.78rem;
               font-weight: 600; color: #fff; margin-right: 6px; }
  .lca-muted { color: #5B6573; font-size: 0.88rem; }
  /* tabs scroll instead of wrapping on small screens */
  [data-baseweb="tab-list"] { overflow-x: auto; scrollbar-width: thin; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""", unsafe_allow_html=True)


# ============================================================ formatting
def minutes_to_dd_hh_mm_ss(minutes):
    """Convert minutes to dd:hh:mm:ss format"""
    if minutes is None or minutes < 0:
        return "00:00:00:00"
    total_seconds = int(minutes * 60)
    days = total_seconds // (24 * 3600)
    remaining_seconds = total_seconds % (24 * 3600)
    hours = remaining_seconds // 3600
    remaining_seconds %= 3600
    mins = remaining_seconds // 60
    secs = remaining_seconds % 60
    return f"{days:02d}:{hours:02d}:{mins:02d}:{secs:02d}"


def format_lead_time(minutes):
    """Format lead time ke dd:hh:mm:ss"""
    if minutes is None:
        return "N/A"
    try:
        minutes_float = float(minutes)
        if minutes_float <= 0:
            return "00:00:00:00"
        return minutes_to_dd_hh_mm_ss(minutes_float)
    except (ValueError, TypeError):
        return "N/A"


def truncate(text, limit):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def esc(text):
    """Escape user/customer-provided text before it goes into HTML."""
    return html.escape("" if text is None else str(text))


def md_escape(text):
    """Escape Markdown control characters in user-provided text."""
    return "".join("\\" + ch if ch in "\\`*_{}[]()<>#+-.!|~" else ch for ch in str(text))


def style_chart(fig, height=320):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        title=dict(font=dict(size=15), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        font=dict(size=12),
    )
    return fig


def show_chart(fig, height=320):
    st.plotly_chart(style_chart(fig, height), use_container_width=True,
                    config={"displaylogo": False, "responsive": True})


# ============================================================ authentication
def _current_user():
    user = st.session_state.get("user")
    if user and not auth.session_still_valid(user):
        st.session_state.pop("user", None)
        st.session_state["auth_notice"] = "Your session ended because the account changed. Please sign in again."
        return None
    return user


def _auth_header(title, subtitle):
    st.markdown(f'<div class="lca-mark" aria-hidden="true">LC</div>'
                f'<p class="lca-auth-title">{esc(title)}</p>'
                f'<p class="lca-auth-sub">{esc(subtitle)}</p>', unsafe_allow_html=True)


def render_login():
    with st.container(key="auth_card"):
        _auth_header(APP_NAME, "Sign in to continue")
        notice = st.session_state.pop("auth_notice", None)
        if notice:
            st.info(notice)
        if auth.count_users() == 0:
            st.warning("No accounts exist yet. Create one on the server with "
                       "`python manage_users.py create --email you@company.com`.")
        with st.form("login_form", border=False):
            email = st.text_input("Email", autocomplete="email", key="login_email")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            if not email.strip() or not password:
                st.error("Enter your email and password.")
            else:
                with st.spinner("Signing in…"):
                    result = auth.authenticate(email, password)
                if result.status == auth.LoginResult.OK:
                    st.session_state["user"] = result.user
                    st.rerun()
                elif result.status == auth.LoginResult.LOCKED:
                    st.error("Too many failed attempts. Wait 15 minutes or reset your password.")
                else:
                    st.error("The email or password is incorrect.")
        if st.button("Forgot password?", type="tertiary"):
            st.session_state["auth_view"] = "forgot"
            st.rerun()


def render_forgot():
    with st.container(key="auth_card"):
        _auth_header("Reset your password",
                     "Enter the email you sign in with. We'll send a link to choose a new password.")
        if st.session_state.get("reset_requested_for"):
            st.success(f"If an account exists for {md_escape(st.session_state['reset_requested_for'])}, "
                       f"a reset link is on its way. The link works once and expires in "
                       f"{auth.reset_ttl_minutes()} minutes.")
            st.caption("Didn't get it? Check your spam folder, or ask your administrator to confirm "
                       "the address on your account.")
        else:
            with st.form("forgot_form", border=False):
                email = st.text_input("Email", autocomplete="email")
                submitted = st.form_submit_button("Send reset link", type="primary", use_container_width=True)
            if submitted:
                if not auth.is_valid_email(email):
                    st.error("Enter a valid email address.")
                else:
                    with st.spinner("Sending…"):
                        auth.request_password_reset(email)
                    st.session_state["reset_requested_for"] = auth.normalise_email(email)
                    st.rerun()
        if st.button("Back to sign in", type="tertiary"):
            st.session_state.pop("reset_requested_for", None)
            st.session_state["auth_view"] = "login"
            st.rerun()


def render_reset(token):
    with st.container(key="auth_card"):
        if st.session_state.get("reset_done"):
            _auth_header("Password updated", "You can now sign in with your new password.")
            if st.button("Go to sign in", type="primary", use_container_width=True):
                st.session_state.pop("reset_done", None)
                st.session_state["auth_view"] = "login"
                st.query_params.clear()
                st.rerun()
            return

        status = auth.check_reset_token(token)
        if status != auth.TokenStatus.VALID:
            messages = {
                auth.TokenStatus.EXPIRED: ("This link has expired", "Reset links are valid for a limited time. Request a new one."),
                auth.TokenStatus.USED: ("This link was already used", "Each reset link works once. Request a new one if you still need it."),
            }
            title, sub = messages.get(status, ("This link is not valid",
                                               "Check that you opened the full link from the email, or request a new one."))
            _auth_header(title, sub)
            if st.button("Request a new link", type="primary", use_container_width=True):
                st.query_params.clear()
                st.session_state["auth_view"] = "forgot"
                st.rerun()
            return

        _auth_header("Choose a new password",
                     f"Use at least {auth.PASSWORD_MIN_LENGTH} characters. A short phrase is easier to remember.")
        with st.form("reset_form", border=False):
            new_pw = st.text_input("New password", type="password", autocomplete="new-password")
            confirm = st.text_input("Confirm new password", type="password", autocomplete="new-password")
            submitted = st.form_submit_button("Update password", type="primary", use_container_width=True)
        if submitted:
            if new_pw != confirm:
                st.error("The passwords don't match.")
                return
            with st.spinner("Updating…"):
                status, problems = auth.reset_password(token, new_pw)
            if problems:
                st.error(" ".join(problems))
            elif status == auth.TokenStatus.VALID:
                st.session_state["reset_done"] = True
                st.session_state.pop("user", None)
                st.rerun()
            else:
                st.error("This link can no longer be used. Request a new one.")


def render_topbar(user):
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([3, 3, 1], vertical_alignment="center")
        c1.markdown(f'<span class="lca-brand">{APP_NAME}</span>', unsafe_allow_html=True)
        c2.markdown(f'<div class="lca-user">{esc(user.get("name") or user["email"])}'
                    f'{" · " + esc(user["email"]) if user.get("name") else ""}</div>',
                    unsafe_allow_html=True)
        if c3.button("Sign out", type="tertiary", key="sign_out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state["auth_notice"] = "You have signed out."
            st.rerun()


# ============================================================ data loading
def _source_key(source):
    if source is None:
        return None
    if isinstance(source, str):
        return f"path:{source}"
    return f"upload:{getattr(source, 'file_id', None) or source.name}:{source.size}"


def load_excel(source, required_columns):
    """Read an uploaded workbook once per file; returns (df, error message)."""
    key = _source_key(source)
    cache = st.session_state.setdefault("_file_cache", {})
    if key in cache:
        return cache[key]
    try:
        if not isinstance(source, str):
            source.seek(0)
        df = pd.read_excel(source)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            out = (None, f"Missing column{'s' if len(missing) > 1 else ''}: {', '.join(missing)}. "
                         f"Found: {', '.join(map(str, df.columns[:8]))}{'…' if len(df.columns) > 8 else ''}")
        elif df.empty:
            out = (None, "The sheet has no rows.")
        else:
            out = (df, None)
    except Exception as e:
        log.warning("Could not read workbook: %s", e)
        out = (None, "This file could not be read as an Excel workbook (.xlsx).")
    cache[key] = out
    return out


def _describe_raw(df):
    tickets = df['Ticket Number'].nunique()
    dates = pd.to_datetime(df['Message Date'], errors='coerce').dropna()
    span = f" · {dates.min():%d %b %Y} – {dates.max():%d %b %Y}" if not dates.empty else ""
    return f"{len(df):,} messages · {tickets:,} tickets{span}", tickets


# ============================================================ upload screen
def main_interface():
    st.markdown("## New analysis")
    st.markdown('<p class="lca-muted">Upload the live chat export and the complaint list for the same '
                'period. Files are used only for this session and are not kept on the server.</p>',
                unsafe_allow_html=True)

    use_sample = st.toggle("Use sample data (40 synthetic tickets, no real customers)", key="use_sample")

    raw_df = complaint_df = None
    total_tickets = 0
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown("**1. Conversation export**")
            if use_sample:
                raw_source = SAMPLE_RAW
                st.caption(f"Using `{os.path.basename(SAMPLE_RAW)}`")
            else:
                raw_source = st.file_uploader("Conversation export (.xlsx)", type=['xlsx'], key="raw_uploader",
                                              label_visibility="collapsed")
            st.caption("Columns: " + ", ".join(RAW_REQUIRED_COLUMNS))
            if raw_source is not None:
                raw_df, err = load_excel(raw_source, RAW_REQUIRED_COLUMNS)
                if err:
                    st.error(err)
                else:
                    summary, total_tickets = _describe_raw(raw_df)
                    st.success(summary, icon=":material/check_circle:")
    with col2:
        with st.container(border=True):
            st.markdown("**2. Complaint list**")
            if use_sample:
                complaint_source = SAMPLE_COMPLAINT
                st.caption(f"Using `{os.path.basename(SAMPLE_COMPLAINT)}`")
            else:
                complaint_source = st.file_uploader("Complaint list (.xlsx)", type=['xlsx'], key="complaint_uploader",
                                                    label_visibility="collapsed")
            st.caption("Columns: " + ", ".join(COMPLAINT_REQUIRED_COLUMNS))
            if complaint_source is not None:
                complaint_df, err = load_excel(complaint_source, COMPLAINT_REQUIRED_COLUMNS)
                if err:
                    st.error(err)
                else:
                    st.success(f"{len(complaint_df):,} complaint records", icon=":material/check_circle:")
    # keep only the files currently selected in memory
    active = {_source_key(raw_source), _source_key(complaint_source)}
    st.session_state["_file_cache"] = {k: v for k, v in st.session_state.get("_file_cache", {}).items() if k in active}

    if raw_df is not None:
        with st.expander("Preview the first rows"):
            st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
            if complaint_df is not None:
                st.dataframe(complaint_df.head(10), use_container_width=True, hide_index=True)

    ready = raw_df is not None and complaint_df is not None
    max_tickets = None
    if ready:
        scope = st.radio("Tickets to analyze", [f"All {total_tickets:,} tickets", "Only the first N tickets"],
                         horizontal=True, key="scope")
        if scope.startswith("Only"):
            max_tickets = st.number_input("Number of tickets", min_value=1, max_value=max(total_tickets, 1),
                                          value=min(100, total_tickets), step=50)
        if total_tickets > 2000 and max_tickets is None:
            st.caption("Large files can take several minutes. Keep this tab open while it runs.")

    clicked = st.button("Run analysis", type="primary", disabled=not ready,
                        help=None if ready else "Upload both files first")
    if not ready and not use_sample:
        st.caption("No files at hand? Turn on **Use sample data** above, or download the sample files to see the expected format.")
        c1, c2, _ = st.columns([1, 1, 2])
        for col, path in ((c1, SAMPLE_RAW), (c2, SAMPLE_COMPLAINT)):
            with open(path, "rb") as f:
                col.download_button(os.path.basename(path), f.read(), file_name=os.path.basename(path),
                                    icon=":material/download:", key=f"dl_{path}")

    if clicked and ready:
        results, stats, report = run_enhanced_analysis(raw_df, complaint_df, max_tickets)
        if results is not None:
            st.session_state.analysis_complete = True
            st.session_state.analysis_results = results
            st.session_state.analysis_stats = stats
            st.session_state.report_bytes = report
            st.session_state.analysis_meta = {
                "finished_at": datetime.now(),
                "total_tickets": total_tickets,
                "source": "Sample data" if use_sample else getattr(raw_source, "name", ""),
            }
            st.rerun()


def run_enhanced_analysis(raw_df, complaint_df, max_tickets):
    """Run the analysis pipeline; returns (results, stats, excel bytes) or (None, None, None)."""
    progress = st.progress(0, text="Preparing data…")
    complaint_path = report_path = None
    try:
        processed_df = DataPreprocessor().clean_data(raw_df)
        if processed_df is None or processed_df.empty:
            progress.empty()
            st.error("No usable messages were found. Check that **Message**, **Ticket Number** and "
                     "**Message Date** are filled in and that dates are valid.")
            return None, None, None

        # The pipeline reads the complaint list from a path.
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            complaint_path = tmp.name
        complaint_df.to_excel(complaint_path, index=False)
        pipeline = CompleteAnalysisPipeline(complaint_data_path=complaint_path)

        def on_progress(done, total):
            if done == total or done % 5 == 0:
                progress.progress(min(done / total, 1.0) * 0.95, text=f"Analyzing ticket {done:,} of {total:,}…")

        results, stats = pipeline.analyze_all_tickets(processed_df, max_tickets=max_tickets,
                                                      progress_callback=on_progress)
        if not results:
            progress.empty()
            st.error("The analysis produced no results.")
            return None, None, None

        progress.progress(0.97, text="Building the Excel report…")
        report_path = ResultsExporter().export_comprehensive_results(results, stats)
        report = None
        if report_path and os.path.exists(report_path):
            with open(report_path, "rb") as f:
                report = f.read()
        progress.empty()
        return results, stats, report
    except Exception as e:
        progress.empty()
        log.exception("Analysis failed")
        st.error("The analysis stopped because of an unexpected error. Check the file format and try again.")
        with st.expander("Technical details"):
            st.code(f"{e.__class__.__name__}: {e}\n\n{traceback.format_exc()}")
        return None, None, None
    finally:
        # do not leave customer data on the server
        for path in (complaint_path, report_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ============================================================ results screen
def reset_analysis():
    for key in ['analysis_complete', 'analysis_results', 'analysis_stats', 'report_bytes', 'analysis_meta']:
        st.session_state.pop(key, None)


def display_enhanced_results():
    results = st.session_state.analysis_results
    stats = st.session_state.analysis_stats
    meta = st.session_state.get("analysis_meta", {})

    if not stats:
        st.error("No analysis statistics available")
        return

    head, actions = st.columns([3, 2], vertical_alignment="bottom")
    with head:
        st.markdown("## Analysis results")
        analyzed = stats.get('total_tickets', 0)
        total = meta.get("total_tickets") or analyzed
        scope = f"{analyzed:,} of {total:,} tickets" if total and total != analyzed else f"{analyzed:,} tickets"
        finished = meta.get("finished_at")
        st.markdown(f'<p class="lca-muted">{esc(meta.get("source") or "Uploaded data")} · {scope}'
                    f'{" · run at " + finished.strftime("%d %b %Y %H:%M") if finished else ""}</p>',
                    unsafe_allow_html=True)
    with actions:
        a1, a2 = st.columns(2)
        if st.session_state.get("report_bytes"):
            a1.download_button(
                "Excel report", data=st.session_state.report_bytes,
                file_name=f"chat_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary", icon=":material/download:", use_container_width=True)
        else:
            a1.caption("Excel report unavailable")
        if a2.button("New analysis", icon=":material/upload_file:", use_container_width=True):
            reset_analysis()
            st.rerun()

    failed_count = stats.get('failed_analysis', 0)
    tabs = st.tabs(["Overview", "Lead times", "Inquiry types", "Performance", "Tickets",
                    f"Data quality{f' ({failed_count})' if failed_count else ''}"])
    with tabs[0]:
        display_professional_overview_tab(results, stats)
    with tabs[1]:
        display_enhanced_lead_time_tab(results, stats)
    with tabs[2]:
        display_issue_types_tab(results, stats)
    with tabs[3]:
        display_performance_tab(results, stats)
    with tabs[4]:
        display_ticket_explorer_tab(results)
    with tabs[5]:
        display_data_quality_tab(results, stats)


# ------------------------------------------------------------ shared calculations
def _is_valid_lead_time(value):
    """Check if lead time value is valid untuk perhitungan"""
    if value is None or value == 'N/A':
        return False
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


def _final_lead_time_minutes(result):
    """Final reply lead time in minutes (complaint lead times are recorded in days)."""
    if result.get('final_issue_type') == 'complaint':
        days = result.get('final_reply_lead_time_days')
        return float(days) * 24 * 60 if _is_valid_lead_time(days) else None
    minutes = result.get('final_reply_lead_time_minutes')
    return float(minutes) if _is_valid_lead_time(minutes) else None


def _first_lead_time_minutes(result):
    minutes = result.get('first_reply_lead_time_minutes')
    return float(minutes) if _is_valid_lead_time(minutes) else None


def calculate_lead_time_stats(results):
    """Calculate lead time statistics dari results"""
    successful = [r for r in results if r['status'] == 'success']
    first = [v for v in (_first_lead_time_minutes(r) for r in successful) if v is not None]
    final = [v for v in (_final_lead_time_minutes(r) for r in successful) if v is not None]
    return {
        'first_lead_times': first,
        'final_lead_times_minutes': final,
        'first_avg_minutes': np.mean(first) if first else 0,
        'final_avg_minutes': np.mean(final) if final else 0,
        'first_samples': len(first),
        'final_samples': len(final),
    }


def _format_lead_time_for_table(result):
    """Format final lead time untuk tabel dalam dd:hh:mm:ss"""
    minutes = _final_lead_time_minutes(result)
    return format_lead_time(minutes) if minutes is not None else "N/A"


def _format_first_lead_time_for_table(result):
    minutes = _first_lead_time_minutes(result)
    return format_lead_time(minutes) if minutes is not None else "N/A"


def _empty(message):
    st.info(message, icon=":material/info:")


# ------------------------------------------------------------ overview
def display_professional_overview_tab(results, stats):
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        _empty("No tickets could be analyzed. See the Data quality tab for the reasons.")
        return
    lt = calculate_lead_time_stats(results)
    eff = stats.get('reply_effectiveness', {})
    n = len(successful)
    leave = eff.get('customer_leave_cases', 0)

    c1, c2, c3, c4 = st.container(key="kpis_overview").columns(4)
    c1.metric("Tickets analyzed", f"{n:,}", help=f"{stats.get('total_tickets', n):,} tickets in scope; "
                                                 f"{stats.get('failed_analysis', 0)} could not be analyzed.")
    c2.metric("Avg first reply", format_lead_time(lt['first_avg_minutes']),
              help=f"dd:hh:mm:ss · {lt['first_samples']} tickets with a first reply")
    c3.metric("Avg final reply", format_lead_time(lt['final_avg_minutes']),
              help=f"dd:hh:mm:ss · {lt['final_samples']} tickets with a final reply")
    c4.metric("Customer left", f"{leave / n * 100:.1f}%", help=f"{leave} tickets ended by Ticket Automation "
                                                                 "because the customer stopped responding")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        dist = stats.get('issue_type_distribution') or {}
        if dist:
            df = pd.DataFrame({'Inquiry type': [k.upper() for k in dist], 'Tickets': list(dist.values())})
            df['Share'] = (df['Tickets'] / df['Tickets'].sum() * 100).round(1)
            order = [t for t in ISSUE_ORDER if t in set(df['Inquiry type'])] + \
                    [t for t in df['Inquiry type'] if t not in ISSUE_ORDER]
            fig = px.bar(df, x='Tickets', y='Inquiry type', orientation='h', color='Inquiry type',
                         color_discrete_map=ISSUE_COLORS, title='Inquiry type mix',
                         text=df.apply(lambda r: f"{r['Tickets']} ({r['Share']}%)", axis=1),
                         category_orders={'Inquiry type': order})
            fig.update_layout(showlegend=False, yaxis_title=None,
                              xaxis_range=[0, df['Tickets'].max() * 1.35])
            fig.update_traces(textposition='outside', cliponaxis=False)
            show_chart(fig, 280)
        else:
            _empty("No inquiry type data available.")
    with col2:
        perf = stats.get('performance_distribution') or {}
        if perf:
            df = pd.DataFrame({'Rating': [k.upper() for k in perf], 'Tickets': list(perf.values())})
            fig = px.bar(df, x='Rating', y='Tickets', color='Rating', color_discrete_map=PERF_COLORS,
                         title='Performance rating', text='Tickets',
                         category_orders={'Rating': [p for p in PERF_ORDER if p in set(df['Rating'])]})
            fig.update_layout(showlegend=False, xaxis_title=None,
                              yaxis_range=[0, df['Tickets'].max() * 1.18])
            fig.update_traces(textposition='outside', cliponaxis=False)
            show_chart(fig, 280)
        else:
            _empty("No performance data available.")

    st.markdown("#### Reply coverage")
    c1, c2, c3 = st.container(key="kpis_coverage").columns(3)
    c1.metric("First reply found", f"{eff.get('first_reply_found_rate', 0) * 100:.1f}%",
              help="Share of analyzed tickets where a first reply was identified")
    c2.metric("Final reply found", f"{eff.get('final_reply_found_rate', 0) * 100:.1f}%",
              help="Share of analyzed tickets where a final (solution) reply was identified")
    c3.metric("Missing required reply", len(_missing_required_replies(successful)),
              help="Serious/complaint tickets without a first reply, or normal tickets without a final "
                   "reply where the customer did not leave")


# ------------------------------------------------------------ lead times
def display_enhanced_lead_time_tab(results, stats):
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        _empty("No analyzed tickets.")
        return
    lt = calculate_lead_time_stats(results)
    c1, c2, c3, c4 = st.container(key="kpis_lead").columns(4)
    c1.metric("First reply average", format_lead_time(lt['first_avg_minutes']))
    c2.metric("Final reply average", format_lead_time(lt['final_avg_minutes']))
    c3.metric("First replies measured", lt['first_samples'])
    c4.metric("Final replies measured", lt['final_samples'])
    st.caption("Lead times use dd:hh:mm:ss (days:hours:minutes:seconds). Complaint final replies use "
               "Lead Time (Solved) from the complaint list.")

    rows = []
    for r in successful:
        rows.append({'Inquiry Type': r['final_issue_type'].upper(),
                     'first': _first_lead_time_minutes(r), 'final': _final_lead_time_minutes(r)})
    df = pd.DataFrame(rows)

    st.markdown("#### By inquiry type")
    breakdown = []
    for issue_type in [t for t in ISSUE_ORDER if t in set(df['Inquiry Type'])] + \
            [t for t in df['Inquiry Type'].unique() if t not in ISSUE_ORDER]:
        sub = df[df['Inquiry Type'] == issue_type]
        first, final = sub['first'].dropna(), sub['final'].dropna()
        breakdown.append({
            'Inquiry Type': issue_type,
            'First Reply Avg': format_lead_time(first.mean()) if len(first) else 'N/A',
            'First Reply Median': format_lead_time(first.median()) if len(first) else 'N/A',
            'Final Reply Avg': format_lead_time(final.mean()) if len(final) else 'N/A',
            'Final Reply Median': format_lead_time(final.median()) if len(final) else 'N/A',
            'Count of First Reply': len(first),
            'Count of Final Reply': len(final),
        })
    st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        first_df = df.dropna(subset=['first'])
        if not first_df.empty:
            fig = px.box(first_df, x='Inquiry Type', y='first', color='Inquiry Type',
                         color_discrete_map=ISSUE_COLORS, title='First reply distribution (minutes)',
                         category_orders={'Inquiry Type': ISSUE_ORDER})
            fig.update_layout(showlegend=False, yaxis_title='Minutes', xaxis_title=None)
            show_chart(fig)
        else:
            _empty("No first reply lead times measured.")
    with col2:
        final_df = df.dropna(subset=['final']).copy()
        if not final_df.empty:
            final_df['value'] = final_df['final'] / 60
            # complaint lead times are days while chat replies are minutes: use a log axis so both stay readable
            wide_range = final_df['final'].max() / max(final_df['final'].min(), 1e-9) > 100
            fig = px.box(final_df, x='Inquiry Type', y='value', color='Inquiry Type', log_y=wide_range,
                         color_discrete_map=ISSUE_COLORS,
                         title=f'Final reply distribution (hours{", log scale" if wide_range else ""})',
                         category_orders={'Inquiry Type': ISSUE_ORDER})
            unit = 'Hours'
            fig.update_layout(showlegend=False, yaxis_title=unit, xaxis_title=None)
            show_chart(fig)
        else:
            _empty("No final reply lead times measured.")

    final_only = df.dropna(subset=['final'])
    if not final_only.empty:
        longest = final_only.loc[final_only['final'].idxmax()]
        fastest = final_only.loc[final_only['final'].idxmin()]
        st.markdown(
            f"Longest final reply: **{longest['Inquiry Type']}**, {format_lead_time(longest['final'])}  \n"
            f"Fastest final reply: **{fastest['Inquiry Type']}**, {format_lead_time(fastest['final'])}")


# ------------------------------------------------------------ inquiry types
def display_issue_types_tab(results, stats):
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        _empty("No analyzed tickets.")
        return
    summary = {}
    for r in successful:
        s = summary.setdefault(r['final_issue_type'].upper(), {'count': 0, 'quality': [], 'first': 0,
                                                                'final': 0, 'leave': 0})
        s['count'] += 1
        s['quality'].append(r['quality_score'])
        s['first'] += bool(r['first_reply_found'])
        s['final'] += bool(r['final_reply_found'])
        s['leave'] += bool(r.get('customer_leave'))
    rows = []
    for issue_type in [t for t in ISSUE_ORDER if t in summary] + [t for t in summary if t not in ISSUE_ORDER]:
        s = summary[issue_type]
        rows.append({
            'Inquiry Type': issue_type,
            'Tickets': s['count'],
            'Share': f"{s['count'] / len(successful) * 100:.1f}%",
            'Avg Quality Score': round(float(np.mean(s['quality'])), 1),
            'First Reply Rate': f"{s['first'] / s['count'] * 100:.1f}%",
            'Final Reply Rate': f"{s['final'] / s['count'] * 100:.1f}%",
            'Customer Left': s['leave'],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={'Avg Quality Score': st.column_config.NumberColumn(format="%.1f / 6")})
    st.caption("Normal: general questions. Serious: issues that need follow-up by another team. "
               "Complaint: tickets matched to the complaint list by phone number.")


# ------------------------------------------------------------ performance
def display_performance_tab(results, stats):
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        _empty("No analyzed tickets.")
        return
    df = pd.DataFrame([{'Inquiry Type': r['final_issue_type'].upper(),
                        'Rating': r['performance_rating'].upper(),
                        'Quality Score': r['quality_score']} for r in successful])
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        counts = df.groupby(['Inquiry Type', 'Rating']).size().reset_index(name='Tickets')
        fig = px.bar(counts, x='Inquiry Type', y='Tickets', color='Rating', barmode='stack',
                     color_discrete_map=PERF_COLORS, title='Rating by inquiry type',
                     category_orders={'Rating': PERF_ORDER, 'Inquiry Type': ISSUE_ORDER})
        fig.update_layout(xaxis_title=None)
        show_chart(fig)
    with col2:
        q = df['Quality Score'].value_counts().reindex(range(0, 7), fill_value=0).reset_index()
        q.columns = ['Quality Score', 'Tickets']
        fig = px.bar(q, x='Quality Score', y='Tickets', title='Quality score (0–6)', text='Tickets',
                     color_discrete_sequence=['#4C78A8'])
        fig.update_layout(xaxis=dict(dtick=1), yaxis_range=[0, max(q['Tickets'].max(), 1) * 1.18])
        fig.update_traces(textposition='outside', cliponaxis=False)
        show_chart(fig)
    st.caption("Quality score: +2 first reply found, +2 final reply found, +1 customer stayed, "
               "+1 reply requirements met.")


# ------------------------------------------------------------ tickets
def display_ticket_explorer_tab(results):
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        _empty("No analyzed tickets.")
        return
    c1, c2, c3 = st.columns([2, 1.3, 1.3])
    query = c1.text_input("Search", placeholder="Ticket number or words in the question", key="tk_q")
    types = c2.multiselect("Inquiry type", ISSUE_ORDER, key="tk_types")
    ratings = c3.multiselect("Rating", PERF_ORDER, key="tk_ratings")
    left_only = st.checkbox("Only tickets where the customer left", key="tk_left")

    q = query.strip().lower()
    filtered = [r for r in successful
                if (not q or q in str(r['ticket_id']).lower() or q in str(r.get('main_question', '')).lower())
                and (not types or r['final_issue_type'].upper() in types)
                and (not ratings or r['performance_rating'].upper() in ratings)
                and (not left_only or r.get('customer_leave'))]
    st.caption(f"{len(filtered):,} of {len(successful):,} tickets")
    if not filtered:
        _empty("No tickets match these filters.")
        return

    # most decision-relevant columns first so they stay visible on narrow screens
    table = pd.DataFrame([{
        'Ticket': str(r['ticket_id']),
        'Inquiry Type': r['final_issue_type'].upper(),
        'Rating': r['performance_rating'].upper(),
        'Final LT': _format_lead_time_for_table(r),
        'Main Question': truncate(r.get('main_question'), 90),
        'First LT': _format_first_lead_time_for_table(r),
        'First Reply': bool(r['first_reply_found']),
        'Final Reply': bool(r['final_reply_found']),
        'Quality': r['quality_score'],
        'Customer Left': bool(r.get('customer_leave')),
    } for r in filtered])
    st.dataframe(table, use_container_width=True, hide_index=True, height=min(38 + 35 * len(table), 420),
                 column_config={'Main Question': st.column_config.TextColumn(width="large"),
                                'Quality': st.column_config.NumberColumn(format="%d / 6")})

    st.markdown("#### Ticket details")
    by_id = {str(r['ticket_id']): r for r in filtered}
    selected = st.selectbox("Ticket", list(by_id.keys()), key="ticket_selector",
                            format_func=lambda t: f"{t} · {by_id[t]['final_issue_type'].upper()} · "
                                                  f"{truncate(by_id[t].get('main_question'), 50)}")
    if selected:
        display_enhanced_ticket_details(by_id[selected])


def _badge(text, color):
    return f'<span class="lca-badge" style="background:{color}">{esc(text)}</span>'


def display_enhanced_ticket_details(result):
    """Detail satu tiket."""
    issue = result['final_issue_type'].upper()
    rating = result['performance_rating'].upper()
    badges = _badge(issue, ISSUE_COLORS.get(issue, '#6B7280')) + _badge(rating, PERF_COLORS.get(rating, '#6B7280'))
    if result.get('customer_leave'):
        badges += _badge("CUSTOMER LEFT", '#6B7280')
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("**Main question**")
    st.markdown(f'<div class="lca-quote">{esc(result.get("main_question"))}</div>', unsafe_allow_html=True)

    answer_rate = (result['answered_pairs'] / result['total_qa_pairs']) * 100 if result['total_qa_pairs'] > 0 else 0
    c1, c2, c3, c4 = st.container(key="kpis_ticket").columns(4)
    c1.metric("Quality score", f"{result['quality_score']}/6")
    c2.metric("Messages", result['total_messages'])
    c3.metric("Q&A pairs answered", f"{result['answered_pairs']}/{result['total_qa_pairs']}",
              help=f"Answer rate {answer_rate:.1f}%")
    c4.metric("Final reply LT", _format_lead_time_for_table(result))

    reply_analysis = result.get('_raw_reply_analysis') or {}
    col1, col2 = st.columns(2, gap="medium")
    for col, key, label in ((col1, 'first_reply', 'First reply'), (col2, 'final_reply', 'Final reply')):
        with col:
            reply = reply_analysis.get(key)
            st.markdown(f"**{label}**")
            if reply:
                st.markdown(f'<div class="lca-quote">{esc(reply.get("message", ""))}</div>', unsafe_allow_html=True)
                if reply.get('lead_time_days'):
                    lead = f"{reply.get('lead_time_days')} days"
                else:
                    lead = format_lead_time(reply.get('lead_time_minutes')) if reply.get('lead_time_minutes') is not None else 'N/A'
                st.caption(f"Time: {reply.get('timestamp', 'N/A')} · Lead time: {lead}"
                           f"{' · ' + str(reply.get('note')) if reply.get('note') else ''}")
            else:
                st.caption("Not found")

    if result.get('_raw_qa_pairs'):
        with st.expander(f"Conversation Q&A pairs ({len(result['_raw_qa_pairs'])})"):
            qa = pd.DataFrame([{
                'Pair': i + 1,
                'Question': pair.get('question', ''),
                'Question Time': str(pair.get('question_time', 'N/A')),
                'Answered': bool(pair.get('is_answered')),
                'Answer': truncate(pair.get('answer') or 'No answer', 200),
                'Answer Time': str(pair.get('answer_time', 'N/A')),
                'Lead Time (min)': pair.get('lead_time_minutes'),
            } for i, pair in enumerate(result['_raw_qa_pairs'])])
            st.dataframe(qa, use_container_width=True, hide_index=True)

    main_issue = reply_analysis.get('main_issue') or {}
    with st.expander("How this ticket was classified"):
        st.markdown(
            f"- Main issue detection score: {esc(main_issue.get('detection_score', result.get('main_issue_detection_score', 'N/A')))}\n"
            f"- Detection reason: {esc(main_issue.get('detection_reason', result.get('main_issue_detection_reason', 'N/A')))}\n"
            f"- Requirement compliant: {reply_analysis.get('requirement_compliant', 'N/A')}\n"
            f"- Question time: {esc(main_issue.get('question_time', result.get('main_question_time', 'N/A')))}\n"
            f"- Analyzed at: {esc(result.get('analysis_timestamp', 'N/A'))}")
        if st.checkbox("Show raw analysis data", key=f"raw_{result['ticket_id']}"):
            st.json(reply_analysis, expanded=False)


# ------------------------------------------------------------ data quality
def _missing_required_replies(successful):
    missing = []
    for r in successful:
        issue_type = r['final_issue_type']
        if issue_type in ['serious', 'complaint'] and not r['first_reply_found']:
            missing.append(r['ticket_id'])
        elif issue_type == 'normal' and not r['final_reply_found'] and not r.get('customer_leave'):
            missing.append(r['ticket_id'])
    return missing


def display_data_quality_tab(results, stats):
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    c1, c2, c3 = st.container(key="kpis_quality").columns(3)
    c1.metric("Analyzed", len(successful))
    c2.metric("Could not be analyzed", len(failed))
    c3.metric("Analysis success rate", f"{(len(successful) / len(results)) * 100:.1f}%" if results else "0%")

    negative = [r['ticket_id'] for r in successful
                if any(isinstance(v, (int, float)) and v < 0 for v in
                       (r.get('first_reply_lead_time_minutes'), r.get('final_reply_lead_time_minutes')))]
    missing = _missing_required_replies(successful)
    checks = pd.DataFrame([
        {'Check': 'Negative lead times', 'Tickets': len(negative),
         'Examples': ', '.join(map(str, negative[:5])) or '-'},
        {'Check': 'Missing required reply', 'Tickets': len(missing),
         'Examples': ', '.join(map(str, missing[:5])) or '-'},
    ])
    st.markdown("#### Checks")
    st.dataframe(checks, use_container_width=True, hide_index=True)

    st.markdown("#### Tickets that could not be analyzed")
    if failed:
        reasons = {}
        for r in failed:
            reasons.setdefault(r.get('failure_reason') or 'Unknown', []).append(str(r['ticket_id']))
        summary = pd.DataFrame([{'Reason': k, 'Tickets': len(v),
                                 'Share': f"{len(v) / len(failed) * 100:.1f}%",
                                 'Examples': ', '.join(v[:3]) + ('…' if len(v) > 3 else '')}
                                for k, v in sorted(reasons.items(), key=lambda kv: -len(kv[1]))])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame([{'Ticket': str(r['ticket_id']), 'Reason': r.get('failure_reason', 'Unknown')}
                                   for r in failed]), use_container_width=True, hide_index=True)
    else:
        st.success("Every ticket in scope was analyzed.", icon=":material/check_circle:")

    with st.expander("Technical details"):
        if successful:
            st.json({k: str(v) for k, v in successful[0].items() if not k.startswith('_raw')}, expanded=False)
        st.caption(f"Python {sys.version.split()[0]} · pandas {pd.__version__} · Streamlit {st.__version__}")


# ============================================================ entry point
def main():
    auth.init_db()
    token = st.query_params.get("reset_token")
    if token:
        render_reset(token)
        return

    user = _current_user()
    if not user:
        if st.session_state.get("auth_view") == "forgot":
            render_forgot()
        else:
            render_login()
        return

    render_topbar(user)
    if not ANALYSIS_AVAILABLE:
        st.error(f"The analysis module could not be loaded: {ANALYSIS_IMPORT_ERROR}. "
                 "Check that Chat_Analyzer_System.py is present and dependencies are installed.")
        st.stop()
    if st.session_state.get('analysis_complete', False):
        display_enhanced_results()
    else:
        main_interface()


if __name__ == "__main__":
    main()
