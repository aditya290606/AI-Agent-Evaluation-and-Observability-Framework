"""
Environment security validator and authorization gate for AgentPulse.

Ensures that the framework only runs when a valid, authorized `.env` file
is present in the project root directory. If cloned without the `.env` file,
execution is safely halted with clear instructions.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Required security and configuration keys
REQUIRED_KEYS = [
    "AGENTPULSE_AUTH_KEY",
    "AGENTPULSE_API_KEY",
    "ANTHROPIC_API_KEY",
    "DATABASE_URL",
]

PLACEHOLDER_PREFIXES = ("your-", "your_", "yourapi", "placeholder", "todo", "change_me")


def check_environment(env_file_path: Optional[Path] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify that a valid, authorized .env file is present and populated.

    Returns:
        (is_valid, reason_message, status_details)
    """
    target_path = env_file_path or ENV_PATH

    # 1. Check physical file existence
    if not target_path.exists():
        return (
            False,
            f"Missing required configuration file: '{target_path.name}' was not found in the project root directory.",
            {"file_exists": False, "missing_keys": REQUIRED_KEYS, "path": str(target_path)},
        )

    # 2. Parse the environment file directly to verify it contains the required keys
    from dotenv import dotenv_values
    file_vars = dotenv_values(dotenv_path=target_path)

    # 3. Check required variables
    missing_keys = []
    placeholder_keys = []

    for key in REQUIRED_KEYS:
        val = file_vars.get(key)
        if val is None:
            missing_keys.append(key)
        else:
            val_clean = str(val).strip().strip('"').strip("'")
            if not val_clean:
                missing_keys.append(key)
            elif any(val_clean.lower().startswith(p) for p in PLACEHOLDER_PREFIXES):
                placeholder_keys.append(key)

    if missing_keys:
        return (
            False,
            f"Incomplete .env file: missing required security keys: {', '.join(missing_keys)}",
            {"file_exists": True, "missing_keys": missing_keys, "placeholder_keys": placeholder_keys, "path": str(target_path)},
        )

    if placeholder_keys:
        return (
            False,
            f"Unconfigured .env file: placeholders detected for: {', '.join(placeholder_keys)}. Paste authorized keys.",
            {"file_exists": True, "missing_keys": [], "placeholder_keys": placeholder_keys, "path": str(target_path)},
        )

    # 4. Populate os.environ with validated file variables
    load_dotenv(dotenv_path=target_path, override=True)

    return (
        True,
        "Authorized environment configuration verified.",
        {"file_exists": True, "missing_keys": [], "placeholder_keys": [], "path": str(target_path)},
    )


def enforce_environment(env_file_path: Optional[Path] = None) -> None:
    """CLI enforcement gate. Halts program with exit code 1 if unauthorized."""
    is_valid, message, details = check_environment(env_file_path)
    if not is_valid:
        # Allow automated CI pipelines (GitHub Actions, etc.) to run with mock credentials
        if os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
            print("\n" + "=" * 80)
            print(" [AGENTPULSE CI MODE] Automated CI pipeline environment detected.")
            print(" Using mock environment credentials for automated evaluation.")
            print("=" * 80 + "\n")
            os.environ.setdefault("AGENTPULSE_AUTH_KEY", "ap_sec_ci_pipeline_authorized_9f83a4c172e")
            os.environ.setdefault("AGENTPULSE_API_KEY", "ap_live_ci_pipeline_telemetry_7c8d9e2f1a0")
            os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-api03-ci-mock-testing-credentials")
            os.environ.setdefault("DATABASE_URL", "sqlite:///./eval_framework.db")
            os.environ.setdefault("USE_MOCK_LLM", "true")
            return

        print("\n" + "=" * 80)
        print(" [AGENTPULSE SECURITY LOCK] AUTHORIZED .ENV FILE REQUIRED")
        print("=" * 80)
        print(f"\n  Status  : ACCESS DENIED")
        print(f"  Reason  : {message}")
        print(f"  Target  : {details.get('path', str(ENV_PATH))}")
        print("\n  [Action Required]")
        print("  This repository is protected. To run the evaluation framework:")
        print("    1. Paste the authorized '.env' file into the project root directory:")
        print(f"       -> {PROJECT_ROOT}")
        print("    2. Re-run your command.\n")
        print("  NOTICE: Never commit or push the '.env' file to GitHub.")
        print("=" * 80 + "\n")
        sys.exit(1)


def render_streamlit_lock_if_unauthorized() -> None:
    """Streamlit enforcement gate. Displays an executive lock screen and stops app execution."""
    import streamlit as st

    is_valid, message, details = check_environment()
    if not is_valid:
        st.markdown(
            """
            <style>
            .stApp {
                background: radial-gradient(ellipse at 50% -20%, #1a0f2e 0%, #080c14 70%, #05080e 100%) fixed !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="max-width: 780px; margin: 60px auto; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(244, 63, 94, 0.35); border-radius: 16px; padding: 36px 40px; box-shadow: 0 0 50px rgba(244, 63, 94, 0.15); backdrop-filter: blur(20px);">
                <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 20px;">
                    <div style="width: 48px; height: 48px; border-radius: 12px; background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.3); display: flex; align-items: center; justify-content: center; font-size: 24px;">
                        🔒
                    </div>
                    <div>
                        <div style="font-size: 11px; font-weight: 700; color: #fb7185; letter-spacing: 0.14em; text-transform: uppercase;">
                            SECURITY AUTHORIZATION REQUIRED
                        </div>
                        <div style="font-size: 24px; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                            AgentPulse Console Locked
                        </div>
                    </div>
                </div>

                <div style="background: rgba(244, 63, 94, 0.08); border-left: 4px solid #f43f5e; padding: 14px 18px; border-radius: 6px; margin-bottom: 24px; font-size: 13.5px; color: #fecdd3;">
                    {message}
                </div>

                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6; margin-bottom: 24px;">
                    This project was cloned from GitHub without the required environment secrets. To unlock and launch the full evaluation console:
                    <ol style="margin-top: 10px; padding-left: 20px; color: #f1f5f9;">
                        <li style="margin-bottom: 6px;">Obtain the authorized <code>.env</code> file from the repository owner.</li>
                        <li style="margin-bottom: 6px;">Paste the <code>.env</code> file directly into the project root: <br/><code style="color:#38bdf8; font-size:12px;">{details.get('path', '.env')}</code></li>
                        <li>Click <strong>Verify & Unlock</strong> below to boot the application.</li>
                    </ol>
                </div>

                <div style="padding: 14px 18px; background: rgba(0, 0, 0, 0.35); border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06); font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #94a3b8; margin-bottom: 24px;">
                    <div style="color: #64748b; margin-bottom: 6px;"># Required credentials checklist</div>
                    <div>{'✅' if details.get('file_exists') else '❌'} .env file present in root</div>
                    <div>{'❌' if 'AGENTPULSE_AUTH_KEY' in details.get('missing_keys', []) else '✅'} AGENTPULSE_AUTH_KEY</div>
                    <div>{'❌' if 'AGENTPULSE_API_KEY' in details.get('missing_keys', []) else '✅'} AGENTPULSE_API_KEY</div>
                    <div>{'❌' if 'ANTHROPIC_API_KEY' in details.get('missing_keys', []) else '✅'} ANTHROPIC_API_KEY</div>
                    <div>{'❌' if 'DATABASE_URL' in details.get('missing_keys', []) else '✅'} DATABASE_URL</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Verify & Unlock", type="primary", use_container_width=True):
                st.rerun()

        st.stop()
