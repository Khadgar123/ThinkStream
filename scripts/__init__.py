"""Project-local command modules.

This file intentionally makes ``scripts`` a regular package so imports such as
``scripts.agent_data`` resolve to this repository before the vendored Verl
checkout's own ``scripts`` package.
"""
