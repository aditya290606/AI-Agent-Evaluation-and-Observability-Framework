from setuptools import setup, find_packages

setup(
    name="agentpulse",
    version="1.0.0",
    description="Lightweight Python AgentPulse SDK for AI Agent Observability & Evaluation",
    packages=find_packages(include=["agentpulse", "agentpulse.*"]),
    python_requires=">=3.8",
    install_requires=[],
)
