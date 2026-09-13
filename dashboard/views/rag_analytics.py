"""
Section 9: RAG Analytics - Retrieval Quality, Context Precision, Faithfulness, Pipeline Flow, and Gap Analysis.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.core.entities import Trace
from src.evaluation.rag_evaluators import extract_retrieval_context, ALL_RAG_EVALUATORS
from dashboard.views.common import steps_to_spans, render_kpi_card, navigate_to, render_section_header


def render_rag_analytics(
    filtered_runs: pd.DataFrame,
    filtered_evals: pd.DataFrame,
    load_steps_fn: Any,
    tc_manager: Any,
):
    render_section_header(
        title="RAG Analytics & Knowledge Grounding",
        subtitle="Deep-dive evaluation into vector/hybrid retrieval accuracy, context precision, answer faithfulness, and hallucination gap detection.",
        breadcrumb="OBSERVABILITY // RAG PIPELINE",
        action_badge="RETRIEVAL QUALITY",
    )

    if filtered_runs.empty:
        st.info("No runs found matching active filters.")
        return

    # Aggregate RAG metrics from filtered_evals if present
    rag_eval_names = ["rag_context_precision", "rag_answer_faithfulness", "keyword_groundedness"]
    rag_evals_df = filtered_evals[filtered_evals["metric_name"].isin(rag_eval_names)] if not filtered_evals.empty else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("RAG Assertions", str(len(rag_evals_df)), subtitle="Evaluated Checks", icon="📚", accent_color="#38bdf8")
    with c2:
        prec_df = rag_evals_df[rag_evals_df["metric_name"] == "rag_context_precision"]
        avg_prec = (prec_df["score"].mean() * 100) if not prec_df.empty else 0.0
        render_kpi_card("Context Precision", f"{avg_prec:.1f}%", subtitle="Relevant Docs Ratio", health="🟢 Healthy" if avg_prec >= 80 else "🟡 Degraded", icon="🎯", accent_color="#10b981")
    with c3:
        faith_df = rag_evals_df[rag_evals_df["metric_name"] == "rag_answer_faithfulness"]
        avg_faith = (faith_df["score"].mean() * 100) if not faith_df.empty else 0.0
        render_kpi_card("Answer Faithfulness", f"{avg_faith:.1f}%", subtitle="Supported by Context", health="🟢 Healthy" if avg_faith >= 85 else "🔴 Critical", icon="⚓", accent_color="#818cf8")
    with c4:
        ground_df = rag_evals_df[rag_evals_df["metric_name"] == "keyword_groundedness"]
        avg_gr = (ground_df["score"].mean() * 100) if not ground_df.empty else 0.0
        render_kpi_card("Groundedness", f"{avg_gr:.1f}%", subtitle="Knowledge Recall", health="🟢 Healthy" if avg_gr >= 80 else "🟡 Degraded", icon="📦", accent_color="#06b6d4")

    st.markdown("<div style='margin-top: 10px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # Select Run for In-Depth RAG Pipeline Inspection
    st.subheader("🔍 Inspect Single Run RAG Pipeline & Telemetry")
    all_run_ids = filtered_runs["run_id"].tolist()
    
    # Pick a preferred RAG run by default if not set by drill-down
    default_rid = st.session_state.get("selected_run_id")
    if not default_rid or default_rid not in all_run_ids:
        rag_candidate_runs = filtered_runs[filtered_runs["task_id"].str.contains("RAG|policy|refund|search|T001|knowledge", case=False, na=False)]
        if not rag_candidate_runs.empty:
            default_rid = rag_candidate_runs.iloc[0]["run_id"]
        else:
            default_rid = all_run_ids[0]

    selected_run_id = st.selectbox(
        "Select Run ID for RAG Deep-Dive:",
        all_run_ids,
        index=all_run_ids.index(default_rid),
        key="rag_run_selector"
    )

    run_row = filtered_runs[filtered_runs["run_id"] == selected_run_id].iloc[0]
    steps = load_steps_fn(int(selected_run_id))
    span_objects = steps_to_spans(steps)

    run_trace = Trace(
        task_id=run_row["task_id"],
        query=run_row["query"],
        final_answer=run_row["final_answer"],
        spans=span_objects,
        is_mock=bool(run_row.get("is_mock", True)),
    )
    run_trace.latency_ms = float(run_row.get("latency_ms") or 0.0)
    run_tc = tc_manager.get_test_case(run_row["task_id"])

    rag_ctx = extract_retrieval_context(run_trace)

    if not rag_ctx or not rag_ctx.has_retrieval:
        st.warning(f"Run #{selected_run_id} does not contain explicit retrieval operations or spans. Select a run with knowledge base retrieval.")
        return

    # 1. RAG Pipeline Flow Diagram
    doc_count_str = f"{len(rag_ctx.retrieved_documents)} Doc{'s' if len(rag_ctx.retrieved_documents) != 1 else ''}"
    with st.container(border=True):
        st.markdown(
            f""" <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;"> 🔄 End-to-End Retrieval Flow </div> <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 12.5px;"> <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 5px 12px; border-radius: 6px; font-weight: 700;">💬 User Query</span> <span style="color: #64748b;">➔</span> <span style="background: rgba(129, 140, 248, 0.12); color: #818cf8; border: 1px solid rgba(129, 140, 248, 0.3); padding: 5px 12px; border-radius: 6px; font-weight: 700;">🔍 Retrieval Query</span> <span style="color: #64748b;">➔</span> <span style="background: rgba(245, 158, 11, 0.12); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); padding: 5px 12px; border-radius: 6px; font-weight: 700;">📄 Retrieved ({doc_count_str})</span> <span style="color: #64748b;">➔</span> <span style="background: rgba(16, 185, 129, 0.12); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); padding: 5px 12px; border-radius: 6px; font-weight: 700;">📦 Context Injection</span> <span style="color: #64748b;">➔</span> <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 5px 12px; border-radius: 6px; font-weight: 700;">🤖 LLM Generation</span> <span style="color: #64748b;">➔</span> <span style="background: rgba(52, 211, 153, 0.12); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); padding: 5px 12px; border-radius: 6px; font-weight: 700;">🎯 Grounded Answer</span> </div> """,
            unsafe_allow_html=True,
        )

    # 2. Evaluate RAG metrics dynamically for this run
    rag_eval_results = []
    for eval_cls in ALL_RAG_EVALUATORS:
        evaluator = eval_cls()
        res = evaluator.evaluate(test_case=run_tc, execution_result=run_row.get("final_answer"), trace=run_trace)
        rag_eval_results.append(res)

    # 3. Summary Metric Cards
    prec_res = next((r for r in rag_eval_results if r.metric_name == "rag_context_precision"), None)
    faith_res = next((r for r in rag_eval_results if r.metric_name == "rag_answer_faithfulness"), None)

    rag_col1, rag_col2, rag_col3, rag_col4 = st.columns(4)
    with rag_col1:
        render_kpi_card("Docs Retrieved", str(rag_ctx.total_documents_retrieved), subtitle="Knowledge Candidates", icon="📄", accent_color="#38bdf8")
    with rag_col2:
        render_kpi_card("Retrieval Latency", f"{rag_ctx.retrieval_latency_ms:.0f} ms", subtitle="Vector / DB Lookup", icon="⏱️", accent_color="#818cf8")
    with rag_col3:
        prec_val = f"{prec_res.score * 100:.0f}%" if prec_res else "N/A"
        prec_health = "🟢 Healthy" if (prec_res and prec_res.passed) else ("🔴 Critical" if prec_res else None)
        render_kpi_card("Context Precision", prec_val, subtitle="Relevant / Fetched", health=prec_health, icon="🎯", accent_color="#10b981")
    with rag_col4:
        faith_val = f"{faith_res.score * 100:.0f}%" if faith_res else "N/A"
        faith_health = "🟢 Healthy" if (faith_res and faith_res.passed) else ("🔴 Critical" if faith_res else None)
        render_kpi_card("Answer Faithfulness", faith_val, subtitle="Context Support", health=faith_health, icon="⚓", accent_color="#06b6d4")

    st.markdown("<div style='margin-top: 14px; margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # 4. Retrieved Documents Cards
    if rag_ctx.retrieved_documents:
        st.markdown("##### 📄 Retrieved Document Details & Telemetry:")
        for d in rag_ctx.retrieved_documents:
            score_disp = f"{d.relevance_score:.3f}" if d.relevance_score is not None else "N/A"
            is_rel = d.relevance_score >= 0.3 if d.relevance_score is not None else True
            badge_color = "#10b981" if is_rel else "#f59e0b"
            badge_text = "RELEVANT" if is_rel else "LOW RELEVANCE"

            with st.container(border=True):
                dc1, dc2, dc3 = st.columns([3, 2, 5])
                with dc1:
                    st.markdown(f"**Rank #{d.rank}:** `{d.title or d.doc_id or 'Document'}`")
                    st.caption(f"Source: `{d.provenance}`")
                with dc2:
                    st.markdown(
                        f""" <span style="background: {badge_color}18; color: {badge_color}; border: 1px solid {badge_color}40; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700;"> {badge_text} ({score_disp}) </span> """,
                        unsafe_allow_html=True,
                    )
                with dc3:
                    st.caption(f'"{d.content[:140]}..."')

    # 5. Dedicated RAG Evaluation Metrics Table
    st.markdown("##### 📊 Dedicated RAG Metric Evaluations:")
    rag_records = []
    for r in rag_eval_results:
        rag_records.append({
            "Metric": r.metric_name,
            "Score": f"{r.score:.2f}",
            "Verdict": "✅ PASS" if r.passed else "❌ FAIL",
            "Threshold": f"{r.threshold:.2f}",
            "Type": r.evaluator_type,
            "Explanation": r.explanation,
        })
    st.dataframe(pd.DataFrame(rag_records), width="stretch")

    # 6. Retrieval Failure / Gap Analysis
    failed_rag = [r for r in rag_eval_results if not r.passed]
    if failed_rag:
        with st.expander("⚠️ Retrieval & Grounding Gap Analysis", expanded=True):
            for fr in failed_rag:
                st.warning(f"**{fr.metric_name}** ({fr.explanation})")
                if fr.evidence and isinstance(fr.evidence, dict):
                    if "irrelevant_docs" in fr.evidence and fr.evidence["irrelevant_docs"]:
                        st.markdown(f"- **Irrelevant Documents Retrieved:** `{fr.evidence['irrelevant_docs']}`")
                    if "missing_tokens" in fr.evidence and fr.evidence["missing_tokens"]:
                        st.markdown(f"- **Missing Expected Info from Context:** `{fr.evidence['missing_tokens']}`")
                    if "ungrounded_tokens" in fr.evidence and fr.evidence["ungrounded_tokens"]:
                        st.markdown(f"- **Ungrounded Claims in Answer:** `{fr.evidence['ungrounded_tokens']}`")
                    if "ungrounded_in_context" in fr.evidence and fr.evidence["ungrounded_in_context"]:
                        st.markdown(f"- **Hallucinated Keywords:** `{fr.evidence['ungrounded_in_context']}`")
    else:
        st.success("✅ **Grounding Verification:** All retrieved documents and generated statements are grounded in factual context.")
