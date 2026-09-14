"""
Database Deduplication & Optimization Script.

Removes all test-polluted runs, dummy agents, duplicate datasets, and duplicate test cases
from eval_framework.db, preserving authentic showcase data and benchmarks.
Reduces database size from 75MB to < 200KB and eliminates dashboard refresh latency.
"""

import os
import shutil
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_framework.db")
BACKUP_PATH = DB_PATH + ".bak"


def clean_database():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found.")
        return

    # 1. Create a backup
    print(f"Creating backup at {BACKUP_PATH}...")
    shutil.copy2(DB_PATH, BACKUP_PATH)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Pre-clean counts
    def get_counts():
        counts = {}
        for tbl in ["runs", "steps", "eval_results", "experiments", "agents", "agent_versions", "test_cases", "datasets"]:
            try:
                counts[tbl] = cursor.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
            except Exception:
                counts[tbl] = 0
        return counts

    pre_counts = get_counts()
    print("Pre-cleanup counts:", pre_counts)

    # 2. Clean dummy agents (created by unit tests like 'Test Support Bot %', 'Toggleable Agent %', 'Sandboxed Agent %')
    cursor.execute("""
        DELETE FROM agents 
        WHERE name LIKE 'Test Support Bot%' 
           OR name LIKE 'Toggleable Agent%'
           OR name LIKE 'Sandboxed Agent%'
           OR agent_id LIKE 'agent_%'
           OR agent_id LIKE 'sb_%'
    """)
    # Remove orphaned agent versions
    cursor.execute("DELETE FROM agent_versions WHERE agent_id NOT IN (SELECT agent_id FROM agents)")

    # 3. Clean duplicate test cases (preserve only golden tasks T001..T015 and CS-001..CS-010)
    cursor.execute("""
        DELETE FROM test_cases 
        WHERE test_id NOT IN (
            'T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008', 'T009', 'T010', 'T011', 'T012', 'T013', 'T014', 'T015',
            'CS-001', 'CS-002', 'CS-003', 'CS-004', 'CS-005', 'CS-006', 'CS-007', 'CS-008', 'CS-009', 'CS-010'
        )
    """)

    # 4. Clean duplicate datasets (keep 1 Customer Support Benchmark and 1 Golden Tasks)
    cursor.execute("""
        DELETE FROM datasets 
        WHERE name IN (
            'Finance Benchmark', 'Membership Suite', 'Original Suite', 'Cloned Suite',
            'Multi-Version Suite', 'Evolution Dataset', 'Automated Test Dataset', 'Cancellation Suite'
        )
    """)
    # If multiple copies of Customer Support Benchmark exist, keep only the latest
    cs_benches = cursor.execute("SELECT id FROM datasets WHERE name = 'Customer Support Benchmark' ORDER BY id DESC").fetchall()
    if len(cs_benches) > 1:
        keep_id = cs_benches[0][0]
        cursor.execute("DELETE FROM datasets WHERE name = 'Customer Support Benchmark' AND id != ?", (keep_id,))

    # Update Customer Support Benchmark test_case_ids to clean list of T001..T015
    import json
    golden_ids = json.dumps([f"T{i:03d}" for i in range(1, 16)])
    cursor.execute("UPDATE datasets SET test_case_ids = ? WHERE name = 'Customer Support Benchmark'", (golden_ids,))

    # 5. Clean experiments:
    # Remove all bloated 'suite_run_%' (which evaluated 1000+ tests each), 'DemoReActAgent_configured_suite_%',
    # and duplicate automated test dataset / cancellation suite experiments.
    cursor.execute("""
        DELETE FROM experiments 
        WHERE name LIKE 'suite_run_%'
           OR name LIKE 'DemoReActAgent_configured_suite_%'
           OR name LIKE 'Evaluation Run: %'
           OR name LIKE 'Toggleable Agent%'
           OR name LIKE 'DemoReActAgent_test_T001_%'
           OR name LIKE 'run_customer_support_agent_%'
    """)

    # Deduplicate experiments sharing the same name
    all_exp_names = [row[0] for row in cursor.execute("SELECT DISTINCT name FROM experiments").fetchall()]
    for exp_name in all_exp_names:
        exp_ids = cursor.execute("SELECT id FROM experiments WHERE name = ? ORDER BY created_at DESC", (exp_name,)).fetchall()
        if len(exp_ids) > 1:
            delete_ids = [e[0] for e in exp_ids[1:]]
            placeholders = ",".join("?" * len(delete_ids))
            cursor.execute(f"DELETE FROM experiments WHERE id IN ({placeholders})", delete_ids)
            cursor.execute(f"DELETE FROM runs WHERE experiment_id IN ({placeholders})", delete_ids)

    # 6. Clean runs: Keep runs belonging to surviving valid experiments or valid agents
    cursor.execute("""
        DELETE FROM runs 
        WHERE experiment_id NOT IN (SELECT id FROM experiments)
           OR agent_name IS NULL
           OR agent_name LIKE 'test_%'
           OR agent_name LIKE 'Toggleable Agent%'
    """)

    # 7. Clean orphaned steps and eval_results
    cursor.execute("DELETE FROM steps WHERE run_id NOT IN (SELECT id FROM runs)")
    cursor.execute("DELETE FROM eval_results WHERE run_id NOT IN (SELECT id FROM runs)")

    conn.commit()

    post_counts = get_counts()
    print("Post-cleanup counts:", post_counts)

    # 8. Rebuild SQLite database file with VACUUM
    print("Running VACUUM to reclaim space...")
    cursor.execute("VACUUM;")
    conn.commit()
    conn.close()

    pre_size = os.path.getsize(BACKUP_PATH) / (1024 * 1024)
    post_size = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Database size reduced from {pre_size:.2f} MB to {post_size:.2f} MB!")


if __name__ == "__main__":
    clean_database()
