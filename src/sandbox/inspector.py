"""Static AST inspector and manifest analyzer for uploaded agent projects.

Performs pure static analysis on Python files and dependency configurations
without executing any user code.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.sandbox.models import (
    ProjectManifest,
    SecurityAlert,
    SecurityCategory,
    SecurityRiskLevel,
    StaticInspectionResult,
)


class ASTSecurityVisitor(ast.NodeVisitor):
    """AST visitor that detects dangerous calls, imports, and system interactions."""

    DANGEROUS_CALLS = {
        "eval": (SecurityRiskLevel.CRITICAL, SecurityCategory.DANGEROUS_EVAL, "Direct use of eval() allows arbitrary code execution"),
        "exec": (SecurityRiskLevel.CRITICAL, SecurityCategory.DANGEROUS_EVAL, "Direct use of exec() allows arbitrary code execution"),
        "__import__": (SecurityRiskLevel.HIGH, SecurityCategory.PROCESS_SPAWN, "Dynamic module import via __import__"),
        "compile": (SecurityRiskLevel.MEDIUM, SecurityCategory.DANGEROUS_EVAL, "Dynamic code compilation via compile()"),
    }

    DANGEROUS_MODULE_CALLS = {
        ("os", "system"): (SecurityRiskLevel.CRITICAL, SecurityCategory.PROCESS_SPAWN, "Host command execution via os.system()"),
        ("os", "popen"): (SecurityRiskLevel.CRITICAL, SecurityCategory.PROCESS_SPAWN, "Host command execution via os.popen()"),
        ("os", "spawn"): (SecurityRiskLevel.HIGH, SecurityCategory.PROCESS_SPAWN, "Process spawn via os.spawn"),
        ("os", "execl"): (SecurityRiskLevel.CRITICAL, SecurityCategory.PROCESS_SPAWN, "Process replacement via os.execl"),
        ("os", "execv"): (SecurityRiskLevel.CRITICAL, SecurityCategory.PROCESS_SPAWN, "Process replacement via os.execv"),
        ("subprocess", "Popen"): (SecurityRiskLevel.HIGH, SecurityCategory.PROCESS_SPAWN, "Subprocess invocation via subprocess.Popen"),
        ("subprocess", "run"): (SecurityRiskLevel.HIGH, SecurityCategory.PROCESS_SPAWN, "Subprocess execution via subprocess.run"),
        ("subprocess", "call"): (SecurityRiskLevel.HIGH, SecurityCategory.PROCESS_SPAWN, "Subprocess execution via subprocess.call"),
        ("subprocess", "check_output"): (SecurityRiskLevel.HIGH, SecurityCategory.PROCESS_SPAWN, "Subprocess execution via subprocess.check_output"),
        ("shutil", "rmtree"): (SecurityRiskLevel.MEDIUM, SecurityCategory.FILESYSTEM_ACCESS, "Recursive directory deletion via shutil.rmtree"),
        ("socket", "socket"): (SecurityRiskLevel.HIGH, SecurityCategory.NETWORK_ACCESS, "Raw network socket creation"),
        ("ctypes", "cdll"): (SecurityRiskLevel.CRITICAL, SecurityCategory.PROCESS_SPAWN, "Direct C library loading / memory manipulation"),
    }

    SENSITIVE_PATHS = [
        "/etc/passwd",
        "/etc/shadow",
        "~/.ssh",
        "~/.aws",
        ".env",
        "/proc",
        "C:\\Windows\\System32",
    ]

    def __init__(self, relative_path: str):
        self.relative_path = relative_path
        self.alerts: List[SecurityAlert] = []
        self.imported_modules: Set[str] = set()
        self.required_env_vars: Set[str] = set()
        self.entry_functions: List[str] = []
        self.entry_classes: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            mod = alias.name.split(".")[0]
            self.imported_modules.add(mod)
            if mod in ("ctypes", "pty", "posix"):
                self.alerts.append(
                    SecurityAlert(
                        rule_id="SUSPICIOUS_IMPORT",
                        category=SecurityCategory.PROCESS_SPAWN,
                        risk_level=SecurityRiskLevel.HIGH,
                        message=f"Import of low-level / native system module: '{alias.name}'",
                        file_path=self.relative_path,
                        line_number=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            mod = node.module.split(".")[0]
            self.imported_modules.add(mod)
            if mod in ("ctypes", "pty", "posix"):
                self.alerts.append(
                    SecurityAlert(
                        rule_id="SUSPICIOUS_IMPORT",
                        category=SecurityCategory.PROCESS_SPAWN,
                        risk_level=SecurityRiskLevel.HIGH,
                        message=f"Import from low-level / native system module: '{node.module}'",
                        file_path=self.relative_path,
                        line_number=node.lineno,
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Direct function calls: eval(), exec(), etc.
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.DANGEROUS_CALLS:
                risk, cat, msg = self.DANGEROUS_CALLS[func_name]
                self.alerts.append(
                    SecurityAlert(
                        rule_id=f"DANGEROUS_CALL_{func_name.upper()}",
                        category=cat,
                        risk_level=risk,
                        message=msg,
                        file_path=self.relative_path,
                        line_number=node.lineno,
                    )
                )
            # Check open() calls for sensitive paths
            if func_name == "open" and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    val = first_arg.value
                    for sensitive in self.SENSITIVE_PATHS:
                        if sensitive in val:
                            self.alerts.append(
                                SecurityAlert(
                                    rule_id="SENSITIVE_FILE_ACCESS",
                                    category=SecurityCategory.CREDENTIAL_ACCESS,
                                    risk_level=SecurityRiskLevel.CRITICAL,
                                    message=f"Attempted access to sensitive path: '{val}'",
                                    file_path=self.relative_path,
                                    line_number=node.lineno,
                                )
                            )

        # Attribute calls: os.system(), subprocess.run(), os.environ.get(), etc.
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                module_or_obj = node.func.value.id
                key = (module_or_obj, attr_name)
                if key in self.DANGEROUS_MODULE_CALLS:
                    risk, cat, msg = self.DANGEROUS_MODULE_CALLS[key]
                    self.alerts.append(
                        SecurityAlert(
                            rule_id=f"DANGEROUS_API_{module_or_obj}_{attr_name}",
                            category=cat,
                            risk_level=risk,
                            message=msg,
                            file_path=self.relative_path,
                            line_number=node.lineno,
                        )
                    )

                # Detect os.environ.get("KEY") or os.getenv("KEY")
                if (module_or_obj == "os" and attr_name == "getenv") or (module_or_obj == "environ" and attr_name == "get"):
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        self.required_env_vars.add(node.args[0].value)

            elif isinstance(node.func.value, ast.Attribute):
                # e.g. os.environ.get()
                if node.func.value.attr == "environ" and attr_name in ("get", "__getitem__"):
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        self.required_env_vars.add(node.args[0].value)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        # Detect os.environ["KEY"]
        if isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            slice_val = getattr(node.slice, "value", node.slice)
            if isinstance(slice_val, ast.Constant) and isinstance(slice_val.value, str):
                self.required_env_vars.add(slice_val.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Identify agent function entry points
        if not node.name.startswith("_"):
            self.entry_functions.append(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # Identify agent classes
        if not node.name.startswith("_"):
            self.entry_classes.append(node.name)
        self.generic_visit(node)


class ProjectInspector:
    """Performs static AST inspection, dependency parsing, framework detection, and security assessment."""

    FRAMEWORK_SIGNATURES = {
        "langchain": ["langchain", "langchain_core", "langchain_community"],
        "langgraph": ["langgraph"],
        "crewai": ["crewai"],
        "autogen": ["autogen", "pyautogen"],
        "openai_swarm": ["swarm"],
        "llamaindex": ["llama_index"],
        "custom": [],
    }

    ENTRY_POINT_CANDIDATES = [
        "agent.py",
        "demo_agent.py",
        "main.py",
        "app.py",
        "run.py",
        "bot.py",
        "custom_agent_template.py",
        "src/agent.py",
        "src/main.py",
    ]

    def inspect_directory(self, root_path: Path) -> StaticInspectionResult:
        """Statically inspect an unzipped project workspace."""
        root = Path(root_path).resolve()
        alerts: List[SecurityAlert] = []
        all_imports: Set[str] = set()
        required_env_vars: Set[str] = set()
        detected_framework = "custom"
        entry_point: str | None = None
        entry_symbol: str | None = None
        dependencies: List[str] = []

        # 1. Parse dependencies from requirements.txt / pyproject.toml / Pipfile
        dependencies.extend(self._extract_dependencies(root))

        # 2. Iterate over all Python files and run AST visitor
        all_py_paths = list(root.glob("**/*.py"))
        # Exclude pycache and virtualenv
        python_files = [p for p in all_py_paths if "__pycache__" not in p.parts and ".venv" not in p.parts and "venv" not in p.parts]
        total_loc = 0

        # Sort python files so candidate entry points are inspected first
        python_files.sort(key=lambda p: (0 if p.name in ["agent.py", "demo_agent.py", "main.py", "app.py"] else 1, str(p)))

        candidate_entry_functions: Dict[str, List[str]] = {}
        candidate_entry_classes: Dict[str, List[str]] = {}
        rel_py_files: List[str] = []

        for py_path in python_files:
            rel_path = str(py_path.relative_to(root)).replace("\\", "/")
            rel_py_files.append(rel_path)
            try:
                content = py_path.read_text(encoding="utf-8", errors="replace")
                total_loc += len(content.splitlines())
                tree = ast.parse(content, filename=str(py_path))
            except Exception as e:
                alerts.append(
                    SecurityAlert(
                        rule_id="PARSE_ERROR",
                        category=SecurityCategory.SUSPICIOUS_OPERATIONS,
                        risk_level=SecurityRiskLevel.LOW,
                        message=f"Failed to parse Python syntax: {e}",
                        file_path=rel_path,
                    )
                )
                continue

            visitor = ASTSecurityVisitor(rel_path)
            visitor.visit(tree)

            alerts.extend(visitor.alerts)
            all_imports.update(visitor.imported_modules)
            required_env_vars.update(visitor.required_env_vars)

            if visitor.entry_functions:
                candidate_entry_functions[rel_path] = visitor.entry_functions
            if visitor.entry_classes:
                candidate_entry_classes[rel_path] = visitor.entry_classes

        # 3. Detect framework from imports and dependencies
        combined_pkgs = all_imports.union({d.split("==")[0].split(">=")[0].strip().lower() for d in dependencies})
        for framework, signatures in self.FRAMEWORK_SIGNATURES.items():
            if any(sig.lower() in combined_pkgs for sig in signatures):
                detected_framework = framework
                break

        # 4. Detect entry point and entry symbol
        for candidate in self.ENTRY_POINT_CANDIDATES:
            cand_norm = candidate.replace("\\", "/")
            if cand_norm in rel_py_files:
                entry_point = cand_norm
                break
            # Check basename match
            for rpf in rel_py_files:
                if Path(rpf).name == candidate:
                    entry_point = rpf
                    break
            if entry_point:
                break

        if not entry_point and rel_py_files:
            entry_point = rel_py_files[0]

        # Find entry symbol in candidate
        if entry_point:
            funcs = candidate_entry_functions.get(entry_point, [])
            classes = candidate_entry_classes.get(entry_point, [])

            # Prefer standard names: run_agent_mock, run, execute, invoke, main, Agent classes
            preferred_names = ["run_agent_mock", "run_agent", "run", "execute", "invoke", "call", "step", "main"]
            for pref in preferred_names:
                if pref in funcs:
                    entry_symbol = pref
                    break

            if not entry_symbol and classes:
                entry_symbol = classes[0]
            elif not entry_symbol and funcs:
                entry_symbol = funcs[0]

        manifest = ProjectManifest(
            entry_point=entry_point or "agent.py",
            entry_symbol=entry_symbol or "run",
            framework=detected_framework,
            dependencies=dependencies,
            required_env_vars=sorted(list(required_env_vars)),
            total_files=len(rel_py_files),
            python_files=rel_py_files,
            description=f"Detected {detected_framework} agent project ({len(rel_py_files)} Python files, {total_loc} LOC).",
        )

        # 5. Check for sensitive keys in env vars
        common_api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"]
        for key in common_api_keys:
            if key in required_env_vars:
                # Flag as sensitive env requirement
                alerts.append(
                    SecurityAlert(
                        rule_id="ENV_KEY_REQUIRED",
                        category=SecurityCategory.CREDENTIAL_ACCESS,
                        risk_level=SecurityRiskLevel.LOW,
                        message=f"Agent requires API credential env var: '{key}' (injected strictly via sandbox config)",
                    )
                )

        # 6. Compute overall risk level and decision
        has_critical = any(a.risk_level == SecurityRiskLevel.CRITICAL for a in alerts)
        has_high = any(a.risk_level == SecurityRiskLevel.HIGH for a in alerts)
        has_medium = any(a.risk_level == SecurityRiskLevel.MEDIUM for a in alerts)

        if has_critical:
            overall_risk = SecurityRiskLevel.CRITICAL
            is_safe = False
            rejection_reason = "CRITICAL security risk detected during AST static analysis (e.g. eval/exec, shell spawn, or host credential access)."
        elif has_high:
            overall_risk = SecurityRiskLevel.HIGH
            is_safe = False
            rejection_reason = "HIGH security risk detected during AST analysis (e.g. subprocess or socket creation). Requires explicit override."
        elif has_medium:
            overall_risk = SecurityRiskLevel.MEDIUM
            is_safe = True
            rejection_reason = None
        else:
            overall_risk = SecurityRiskLevel.LOW
            is_safe = True
            rejection_reason = None

        manifest = ProjectManifest(
            entry_point=entry_point or "agent.py",
            entry_symbol=entry_symbol or "run",
            framework=detected_framework,
            dependencies=dependencies,
            required_env_vars=sorted(list(required_env_vars)),
            description=f"Detected {detected_framework} agent project ({len(python_files)} Python files, {total_loc} LOC).",
        )

        return StaticInspectionResult(
            manifest=manifest,
            alerts=alerts,
            overall_risk=overall_risk,
            is_safe_to_execute=is_safe,
            rejection_reason=rejection_reason,
            lines_of_code=total_loc,
            file_count=len(python_files),
        )

    def _extract_dependencies(self, root: Path) -> List[str]:
        """Extract dependency lists from requirements.txt or pyproject.toml."""
        deps: List[str] = []
        req_file = root / "requirements.txt"
        if req_file.is_file():
            try:
                for line in req_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith("#") and not cleaned.startswith("-"):
                        deps.append(cleaned)
            except Exception:
                pass

        pyproject_file = root / "pyproject.toml"
        if pyproject_file.is_file():
            try:
                content = pyproject_file.read_text(encoding="utf-8", errors="replace")
                # Simple regex extraction for dependencies
                matches = re.findall(r'["\']([a-zA-Z0-9_\-\.]+[\>\=\<\~]*[a-zA-Z0-9_\-\.]*)["\']', content)
                for m in matches:
                    if m not in deps and len(m) > 1 and not m.startswith("http"):
                        deps.append(m)
            except Exception:
                pass

        return deps
