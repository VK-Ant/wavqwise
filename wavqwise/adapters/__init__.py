"""
Pre-built adapters for popular PyPI packages.
Plug any library into WavqWise with one line.

from wavqwise.adapters import NixtlaAdapter, DartsAdapter, ProphetAdapter, SkforecastAdapter
"""
from wavqwise.adapters.nixtla import NixtlaAdapter
from wavqwise.adapters.darts import DartsAdapter
from wavqwise.adapters.prophet import ProphetAdapter
from wavqwise.adapters.skforecast import SkforecastAdapter
