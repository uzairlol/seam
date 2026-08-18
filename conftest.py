"""Pytest configuration file to set headless Matplotlib backend for all tests."""

import matplotlib

matplotlib.use("Agg")
