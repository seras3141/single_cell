"""Unit tests for src.visualize.headless_layers."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.visualize.headless_layers import (
    _label_to_rgba,
    build_layer_controls,
    normalize_for_display,
    render_layers,
)


@pytest.fixture
def bf():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (16, 16), dtype=np.uint8)


@pytest.fixture
def mcherry():
    rng = np.random.default_rng(1)
    return rng.integers(0, 4096, (16, 16), dtype=np.uint16)


@pytest.fixture
def mask():
    m = np.zeros((16, 16), dtype=np.uint16)
    m[2:6, 2:6] = 1
    m[10:14, 10:14] = 2
    return m


class TestNormalizeForDisplay:
    def test_output_in_unit_range(self, mcherry):
        result = normalize_for_display(mcherry)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_constant_image_returns_zeros(self):
        constant = np.full((8, 8), 42, dtype=np.uint16)
        result = normalize_for_display(constant)
        np.testing.assert_array_equal(result, np.zeros((8, 8)))


class TestLabelToRgba:
    def test_background_is_transparent(self, mask):
        rgba = _label_to_rgba(mask)
        assert rgba.shape == (16, 16, 4)
        background_alpha = rgba[0, 0, 3]
        assert background_alpha == 0

    def test_foreground_is_opaque(self, mask):
        rgba = _label_to_rgba(mask)
        foreground_alpha = rgba[3, 3, 3]
        assert foreground_alpha == 1


class TestRenderLayers:
    def test_all_layers_present(self, bf, mask, mcherry):
        fig, layer_artists = render_layers(bf, mask=mask, mcherry=mcherry)

        assert layer_artists["bf"] is not None
        assert layer_artists["mask"] is not None
        assert layer_artists["mcherry"] is not None
        assert len(fig.axes[0].images) == 3
        assert fig.axes[0].images[0] is layer_artists["bf"]

    def test_missing_mask_and_mcherry(self, bf):
        fig, layer_artists = render_layers(bf)

        assert layer_artists["bf"] is not None
        assert layer_artists["mask"] is None
        assert layer_artists["mcherry"] is None
        assert len(fig.axes[0].images) == 1

    def test_missing_mcherry_only(self, bf, mask):
        fig, layer_artists = render_layers(bf, mask=mask)

        assert layer_artists["mask"] is not None
        assert layer_artists["mcherry"] is None
        assert len(fig.axes[0].images) == 2


class TestBuildLayerControls:
    pytest.importorskip("ipywidgets")

    def test_one_row_per_non_bf_layer(self, bf, mask, mcherry):
        fig, layer_artists = render_layers(bf, mask=mask, mcherry=mcherry)
        controls = build_layer_controls(layer_artists, fig)
        assert len(controls.children) == 2

    def test_skips_none_layers(self, bf, mask):
        fig, layer_artists = render_layers(bf, mask=mask)
        controls = build_layer_controls(layer_artists, fig)
        assert len(controls.children) == 1

    def test_checkbox_toggles_visibility(self, bf, mask):
        fig, layer_artists = render_layers(bf, mask=mask)
        controls = build_layer_controls(layer_artists, fig)
        checkbox, _slider = controls.children[0].children

        checkbox.value = False

        assert layer_artists["mask"].get_visible() is False

    def test_slider_updates_alpha(self, bf, mask):
        fig, layer_artists = render_layers(bf, mask=mask, mask_alpha=0.5)
        controls = build_layer_controls(layer_artists, fig)
        _checkbox, slider = controls.children[0].children

        slider.value = 0.9

        assert layer_artists["mask"].get_alpha() == pytest.approx(0.9)
