"""The leakage guard must actually fire. If these pass silently, so does a bug."""

import pandas as pd
import pytest

from fpl.features.base import LeakageError, point_in_time, filter_history

DF = pd.DataFrame({
    "season": ["2024-25"] * 5 + ["2025-26"] * 5,
    "gw": [1, 2, 3, 4, 5] * 2,
    "element": [1] * 10,
    "minutes": [90] * 10,
})


def test_filter_excludes_current_and_future_gw():
    h = filter_history(DF, season="2025-26", as_of_gw=3)
    assert h[h.season == "2025-26"].gw.max() == 2
    assert len(h[h.season == "2024-25"]) == 5, "prior seasons are the cold-start prior"


def test_honest_function_passes():
    @point_in_time
    def mean_minutes(df, *, season, as_of_gw):
        return df["minutes"].mean()

    assert mean_minutes(DF, season="2025-26", as_of_gw=3) == 90


def test_output_leak_is_caught():
    @point_in_time
    def sneaky(df, *, season, as_of_gw):
        # Re-joins the full frame, reintroducing the future.
        return DF.copy()

    with pytest.raises(LeakageError, match="leaks"):
        sneaky(DF, season="2025-26", as_of_gw=3)
