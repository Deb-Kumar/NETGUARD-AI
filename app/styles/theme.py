"""Dark navy cybersecurity theme styling and reusable status UI blocks."""

from __future__ import annotations

import html

import streamlit as st

COLOR_BG = "#0A1120"
COLOR_CARD = "#111C33"
COLOR_BORDER = "#1E3A5F"
COLOR_BLUE = "#3B82F6"
COLOR_GREEN = "#10B981"
COLOR_RED = "#EF4444"
COLOR_AMBER = "#F59E0B"
COLOR_TEXT = "#E2E8F0"
COLOR_MUTED = "#94A3B8"


def apply_cyber_theme() -> None:
    """Inject cohesive dark-navy cybersecurity CSS styling."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
            }}
            [data-testid="stSidebar"] {{
                background-color: #070D1A;
                border-right: 1px solid {COLOR_BORDER};
            }}
            .ng-header {{
                background: linear-gradient(135deg, #0F172A 0%, #162447 100%);
                border: 1px solid {COLOR_BORDER};
                border-left: 4px solid {COLOR_BLUE};
                border-radius: 10px;
                padding: 1.1rem 1.4rem;
                margin-bottom: 1.25rem;
            }}
            .ng-header h1 {{
                margin: 0;
                font-size: 1.65rem;
                color: #F8FAFC;
                letter-spacing: 0.02em;
            }}
            .ng-header p {{
                margin: 0.35rem 0 0 0;
                color: {COLOR_MUTED};
                font-size: 0.92rem;
            }}
            .ng-card {{
                background-color: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 1rem 1.15rem;
                margin-bottom: 0.9rem;
                min-height: 96px;
            }}
            .ng-kpi-label {{
                color: {COLOR_MUTED};
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .ng-kpi-value {{
                color: #F8FAFC;
                font-size: clamp(1.05rem, 1.35vw, 1.45rem);
                font-weight: 700;
                margin-top: 0.2rem;
                line-height: 1.25;
                overflow-wrap: break-word;
            }}
            .ng-badge-normal {{
                display: inline-block;
                background: rgba(16, 185, 129, 0.16);
                color: {COLOR_GREEN};
                border: 1px solid {COLOR_GREEN};
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-weight: 600;
                font-size: 0.85rem;
            }}
            .ng-badge-malicious {{
                display: inline-block;
                background: rgba(239, 68, 68, 0.16);
                color: {COLOR_RED};
                border: 1px solid {COLOR_RED};
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-weight: 600;
                font-size: 0.85rem;
            }}
            .ng-badge-warn {{
                display: inline-block;
                background: rgba(245, 158, 11, 0.16);
                color: {COLOR_AMBER};
                border: 1px solid {COLOR_AMBER};
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-weight: 600;
                font-size: 0.85rem;
            }}
            .ng-badge-info {{
                display: inline-block;
                background: rgba(59, 130, 246, 0.16);
                color: {COLOR_BLUE};
                border: 1px solid {COLOR_BLUE};
                border-radius: 999px;
                padding: 0.25rem 0.75rem;
                font-weight: 600;
                font-size: 0.85rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    """Render standardized dark-navy page title banner."""
    safe_title = html.escape(str(title), quote=True)
    safe_subtitle = html.escape(str(subtitle), quote=True)
    st.markdown(
        f"""
        <div class="ng-header">
            <h1>{safe_title}</h1>
            <p>{safe_subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, accent_hex: str = COLOR_BLUE) -> None:
    """Render a compact dark KPI metric card."""
    safe_label = html.escape(str(label), quote=True)
    safe_value = html.escape(str(value), quote=True)
    safe_accent = html.escape(str(accent_hex), quote=True)
    st.markdown(
        f"""
        <div class="ng-card" style="border-top: 3px solid {safe_accent};">
            <div class="ng-kpi-label">{safe_label}</div>
            <div class="ng-kpi-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_setup_message(title: str, command_hint: str) -> None:
    """Display an explicit setup notice when trained artifacts are not yet present."""
    st.warning(
        f"**{title}**\n\n"
        f"No stored results were found for this section. Generate real artifacts first by running:\n"
        f"```powershell\n{command_hint}\n```"
    )
