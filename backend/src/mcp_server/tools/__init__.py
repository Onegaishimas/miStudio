"""
MCP tool modules, one per gated category (Feature 010, BR-5.2).

Each module exposes ``register(mcp, client, settings)``; ``server.py`` calls
it only when the module's category is enabled via ``MCP_TOOL_CATEGORIES``.
"""

from . import admin, discovery, experiments, features, groups, jobs, labeling, steering

# category name → module (registration order = tools/list order)
CATEGORY_MODULES = {
    "read": [discovery, features],
    "groups": [groups],
    "steering": [steering],
    "experiments": [experiments],
    "labeling": [labeling],
    "jobs": [jobs],
    "admin": [admin],
}
