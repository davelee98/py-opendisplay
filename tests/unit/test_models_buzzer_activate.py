"""Tests for BuzzerActivateConfig and helpers."""

from __future__ import annotations

import pytest

from opendisplay.models.buzzer_activate import (
    BuzzerActivateConfig,
    BuzzerPattern,
    BuzzerStep,
    hz_to_index,
    ms_to_units,
)


class TestHzToIndex:
    def test_silence_at_zero(self):
        assert hz_to_index(0) == 0

    def test_silence_at_negative(self):
        assert hz_to_index(-100) == 0

    def test_low_landmark_frequency(self):
        # 400 Hz maps to index 117 (bottom of the firmware's playable window)
        assert hz_to_index(400) == 117

    def test_high_landmark_frequency(self):
        # 12000 Hz maps to index 234 (top of the firmware's playable window)
        assert hz_to_index(12000) == 234

    def test_6200_hz(self):
        assert hz_to_index(6200) == 212

    def test_clamps_above_max(self):
        assert hz_to_index(99999) == 255

    def test_low_non_zero(self):
        # Any positive Hz below the anchor still produces at least index 1 (not 0)
        assert hz_to_index(1) == 1
        # 399 Hz is not "below min" any more; it maps to 117 like 400 Hz
        assert hz_to_index(399) == 117

    def test_concert_a4(self):
        # Reference doc §4.2: nA4 = 440 Hz exactly at idx 120
        assert hz_to_index(440) == 120

    def test_a5_octave(self):
        # nA5 = 880 Hz at idx 144; one octave up is exactly +24 quarter-tones
        assert hz_to_index(880) == 144
        assert hz_to_index(880) - hz_to_index(440) == 24

    def test_landmark_1000_hz(self):
        assert hz_to_index(1000) == 148

    def test_landmark_2000_hz(self):
        assert hz_to_index(2000) == 172

    def test_landmark_523_hz(self):
        assert hz_to_index(523) == 126

    def test_landmark_11840_hz(self):
        assert hz_to_index(11840) == 234

    def test_landmark_21714_hz(self):
        assert hz_to_index(21714) == 255

    @pytest.mark.parametrize("idx", [117, 120, 126, 144, 148, 172, 212, 234])
    def test_round_trip(self, idx):
        # Feeding the firmware's frequency for an index back through hz_to_index
        # recovers that index.
        hz = round(13.75 * 2 ** (idx / 24))
        assert hz_to_index(hz) == idx


class TestMsToUnits:
    def test_minimum_one_unit(self):
        assert ms_to_units(1) == 1

    def test_five_ms_is_one_unit(self):
        assert ms_to_units(5) == 1

    def test_ten_ms_is_two_units(self):
        assert ms_to_units(10) == 2

    def test_rounding(self):
        # 7 ms → round(7/5) = round(1.4) = 1
        assert ms_to_units(7) == 1
        # 8 ms → round(8/5) = round(1.6) = 2
        assert ms_to_units(8) == 2

    def test_max_clamp(self):
        assert ms_to_units(99999) == 255

    def test_zero_returns_minimum(self):
        assert ms_to_units(0) == 1


class TestBuzzerPatternToBytes:
    def test_single_step(self):
        step = BuzzerStep(frequency_index=10, duration_units=20)
        pattern = BuzzerPattern(steps=(step,))
        data = pattern.to_bytes()
        assert data == bytes([1, 10, 20])

    def test_two_steps(self):
        s1 = BuzzerStep(frequency_index=5, duration_units=10)
        s2 = BuzzerStep(frequency_index=200, duration_units=50)
        pattern = BuzzerPattern(steps=(s1, s2))
        data = pattern.to_bytes()
        assert data == bytes([2, 5, 10, 200, 50])


class TestBuzzerActivateConfigToBytes:
    def test_single_tone_wire_format(self):
        config = BuzzerActivateConfig.single_tone(frequency_hz=1000, duration_ms=100, repeats=3)
        data = config.to_bytes()
        # [outer_repeats=3][n_patterns=1][n_steps=1][freq_idx][dur_units]
        assert data[0] == 3  # outer_repeats
        assert data[1] == 1  # 1 pattern
        assert data[2] == 1  # 1 step
        assert len(data) == 5

    def test_single_tone_a4_wire_bytes(self):
        # Reference doc §7.1 worked example: 440 Hz / 200 ms -> freq_idx 120 (0x78),
        # dur_units 40 (0x28).
        config = BuzzerActivateConfig.single_tone(frequency_hz=440, duration_ms=200)
        assert config.to_bytes() == bytes([1, 1, 1, 120, 40])

    def test_silence_tone(self):
        config = BuzzerActivateConfig.single_tone(frequency_hz=0, duration_ms=50)
        data = config.to_bytes()
        assert data[2] == 1  # n_steps
        assert data[3] == 0  # frequency_index = 0 (silence)

    def test_repeats_minimum_one(self):
        config = BuzzerActivateConfig.single_tone(frequency_hz=1000, duration_ms=100, repeats=0)
        assert config.outer_repeats == 1

    def test_default_repeats_is_one(self):
        config = BuzzerActivateConfig.single_tone(frequency_hz=440, duration_ms=200)
        assert config.outer_repeats == 1


class TestBuzzerActivateConfigRoundtrip:
    def test_byte_length_matches_structure(self):
        config = BuzzerActivateConfig.single_tone(frequency_hz=2000, duration_ms=250)
        data = config.to_bytes()
        # [repeats(1)][n_patterns(1)][n_steps(1)][freq(1)][dur(1)] = 5 bytes
        assert len(data) == 5
