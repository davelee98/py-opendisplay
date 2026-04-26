"""Unit tests for partial-rendering module (PartialState, diff, packer, NACK)."""

from __future__ import annotations

import zlib

import pytest

from opendisplay.partial import (
    ERR_ETAG_MISMATCH,
    ERR_MIXED_DATA,
    ERR_PARTIAL_VERSION,
    ERR_SEGMENT_ALIGN,
    ERR_SEGMENT_OOB,
    SEGMENT_HEADER_SIZE,
    SEGMENT_FLAG_COMPRESSED,
    SEGMENT_FLAG_PLANE_1,
    SEGMENT_PIXEL_ALIGN,
    FullImageStrategy,
    PartialState,
    RecursiveBoundingBoxStrategy,
    Segment,
    _generate_etag,
    pack_segments_into_packets,
    parse_nack,
)
from opendisplay.protocol.commands import (
    CHUNK_SIZE,
    ENCRYPTED_CHUNK_SIZE,
    build_direct_write_end_with_etag,
    build_direct_write_partial_start,
    build_partial_data_packet,
)


class TestPartialState:
    def test_roundtrip_empty(self):
        s = PartialState()
        out = PartialState.from_bytes(s.to_bytes())
        assert out == s

    def test_roundtrip_populated(self):
        s = PartialState(
            etag=0xDEADBEEF,
            last_image=bytes(range(256)) * 4,
            width=480,
            height=800,
            bytes_per_pixel=1,
        )
        out = PartialState.from_bytes(s.to_bytes())
        assert out.etag == s.etag
        assert out.last_image == s.last_image
        assert out.width == s.width
        assert out.height == s.height
        assert out.bytes_per_pixel == s.bytes_per_pixel

    def test_bad_magic_rejected(self):
        with pytest.raises(ValueError, match="magic"):
            PartialState.from_bytes(b"XXXX" + b"\x00" * 21)

    def test_truncated_rejected(self):
        with pytest.raises(ValueError):
            PartialState.from_bytes(b"PDST")


class TestParseNack:
    def test_etag_mismatch(self):
        assert parse_nack(b"\xff\x76\x01\x00") == (0x76, ERR_ETAG_MISMATCH)

    def test_mixed_data_on_partial(self):
        assert parse_nack(b"\xff\x77\x02\x00") == (0x77, ERR_MIXED_DATA)

    def test_oob(self):
        assert parse_nack(b"\xff\x77\x03\x00") == (0x77, ERR_SEGMENT_OOB)

    def test_bad_version(self):
        assert parse_nack(b"\xff\x76\x04\x00") == (0x76, ERR_PARTIAL_VERSION)

    def test_segment_align(self):
        assert parse_nack(b"\xff\x77\x05\x00") == (0x77, ERR_SEGMENT_ALIGN)

    def test_not_nack_returns_none(self):
        assert parse_nack(b"\x00\x76") is None
        assert parse_nack(b"\xff\x70\x01") is None       # too short
        assert parse_nack(b"\xff\x70\x01\x00\x00") is None  # too long
        assert parse_nack(b"\xff\x70\x01\x01") is None   # last byte not 0


class TestGenerateEtag:
    def test_nonzero_uint32(self):
        for _ in range(100):
            v = _generate_etag()
            assert 1 <= v <= 0xFFFFFFFF


class TestRecursiveBoundingBoxStrategy:
    @staticmethod
    def _paint_rect(buf: bytearray, width: int, x: int, y: int, w: int, h: int, value: int = 1) -> None:
        for dy in range(h):
            row = (y + dy) * width
            for dx in range(w):
                buf[row + x + dx] = value

    def test_no_change_returns_empty(self):
        buf = bytes(100 * 100)
        segs = RecursiveBoundingBoxStrategy().diff(buf, buf, 100, 100, 1, 4096)
        assert segs == []

    def test_single_pixel_change(self):
        old = bytearray(100 * 100)
        new = bytearray(old)
        new[50 * 100 + 30] = 1
        segs = RecursiveBoundingBoxStrategy().diff(bytes(old), bytes(new), 100, 100, 1, 4096)
        assert len(segs) == 1
        s = segs[0]
        # Bounding box is the single pixel at (30, 50); aligned to 8-pixel
        # boundary becomes x=24, width=8.
        assert s.x == 24 and s.y == 50
        assert s.width == 8 and s.height == 1
        # The changed pixel is at column 30, i.e. index (30 - 24) = 6 in the segment row.
        assert s.pixels == b"\x00\x00\x00\x00\x00\x00\x01\x00"
        assert s.plane == 0

    def test_single_filled_rectangle_returns_aligned_bounds(self):
        w, h = 16, 12
        old = bytearray(w * h)
        new = bytearray(old)
        self._paint_rect(new, w, x=3, y=4, w=5, h=3, value=1)

        segs = RecursiveBoundingBoxStrategy().diff(bytes(old), bytes(new), w, h, 1, 4096)

        assert len(segs) == 1
        seg = segs[0]
        # Original bbox (3, 4, 5, 3): x snaps down to 0, x_end (3+5=8) is already aligned.
        assert (seg.x, seg.y, seg.width, seg.height) == (0, 4, 8, 3)
        assert seg.pixel_count == 24
        # Pixels in cols 3..7 of each row are 1, cols 0..2 are 0.
        expected_row = b"\x00\x00\x00\x01\x01\x01\x01\x01"
        assert seg.pixels == expected_row * 3

    def test_two_disjoint_rectangles_return_two_segments_when_bbox_exceeds_budget(self):
        w, h = 32, 20
        old = bytearray(w * h)
        new = bytearray(old)
        self._paint_rect(new, w, x=1, y=2, w=4, h=3, value=1)
        self._paint_rect(new, w, x=24, y=12, w=5, h=4, value=1)

        # The combined bounding box would be 28x14 = 392 pixels, forcing a split.
        segs = RecursiveBoundingBoxStrategy(min_region_pixels=1).diff(bytes(old), bytes(new), w, h, 1, 64)

        actual = sorted((seg.x, seg.y, seg.width, seg.height) for seg in segs)
        # (1,2,4,3) → x=0, x_end=8  → (0, 2, 8, 3)
        # (24,12,5,4) → x=24, x_end=32 → (24, 12, 8, 4)
        assert actual == [(0, 2, 8, 3), (24, 12, 8, 4)]

    def test_changed_hollow_rectangle_preserves_bounding_box_pixels(self):
        w, h = 16, 10
        old = bytearray(w * h)
        new = bytearray(old)
        self._paint_rect(new, w, x=2, y=2, w=4, h=1, value=1)
        self._paint_rect(new, w, x=2, y=5, w=4, h=1, value=1)
        self._paint_rect(new, w, x=2, y=2, w=1, h=4, value=1)
        self._paint_rect(new, w, x=5, y=2, w=1, h=4, value=1)

        segs = RecursiveBoundingBoxStrategy().diff(bytes(old), bytes(new), w, h, 1, 4096)

        assert len(segs) == 1
        seg = segs[0]
        # Original bbox (2, 2, 4, 4): x snaps down to 0, x_end (2+4=6) snaps up to 8.
        assert (seg.x, seg.y, seg.width, seg.height) == (0, 2, 8, 4)
        # Each row is the original cols 0..7 of the new image at that y.
        # Hollow rect is at cols 2..5, rows 2..5.
        assert seg.pixels == (
            b"\x00\x00\x01\x01\x01\x01\x00\x00"  # y=2: cols 2..5 are top edge
            b"\x00\x00\x01\x00\x00\x01\x00\x00"  # y=3: cols 2 and 5 (sides)
            b"\x00\x00\x01\x00\x00\x01\x00\x00"  # y=4: cols 2 and 5 (sides)
            b"\x00\x00\x01\x01\x01\x01\x00\x00"  # y=5: cols 2..5 are bottom edge
        )

    def test_segments_are_8px_aligned(self):
        # Adversarial rectangles that don't naturally land on 8-pixel boundaries.
        w, h = 64, 32
        old = bytes(w * h)
        new_buf = bytearray(old)
        for x, y, rw, rh in [(1, 1, 3, 3), (13, 5, 5, 4), (37, 20, 11, 7)]:
            self._paint_rect(new_buf, w, x=x, y=y, w=rw, h=rh, value=1)
        segs = RecursiveBoundingBoxStrategy(min_region_pixels=1).diff(
            bytes(old), bytes(new_buf), w, h, 1, max_segment_bytes=128
        )
        assert segs, "expected at least one segment"
        for s in segs:
            assert s.x % SEGMENT_PIXEL_ALIGN == 0, f"x={s.x} not aligned"
            assert s.width % SEGMENT_PIXEL_ALIGN == 0, f"width={s.width} not aligned"
            assert s.x + s.width <= w

    def test_full_change_tiles_image(self):
        # 64x64 with every pixel different; budget 256 bytes
        # → recursion must split until each tile fits.
        w, h = 64, 64
        old = bytes(w * h)
        new = bytes([1] * (w * h))
        segs = RecursiveBoundingBoxStrategy(min_region_pixels=4).diff(
            old, new, w, h, 1, max_segment_bytes=256
        )
        # All segments must be within image bounds and fit budget
        for s in segs:
            assert s.x >= 0 and s.y >= 0
            assert s.x + s.width <= w
            assert s.y + s.height <= h
            assert len(s.pixels) <= 256 or s.width * s.height <= 4
        # Coverage check: union of segments covers every changed pixel
        covered = bytearray(w * h)
        for s in segs:
            for dy in range(s.height):
                for dx in range(s.width):
                    covered[(s.y + dy) * w + (s.x + dx)] = 1
        assert all(b == 1 for b in covered)

    def test_diff_with_fit_uses_wire_fit_predicate(self):
        # The old palette-byte budget would split this full-height change many
        # times. The wire predicate models a mono segment: 512 pixels encode to
        # 64 wire bytes, so the whole changed rectangle should be accepted.
        w, h = 64, 8
        old = bytes(w * h)
        new = bytes([1] * (w * h))

        def mono_wire_fits(seg: Segment) -> bool:
            wire_bytes = (seg.pixel_count + 7) // 8
            return wire_bytes <= 64

        segs = RecursiveBoundingBoxStrategy(min_region_pixels=1).diff_with_fit(old, new, w, h, 1, mono_wire_fits)

        assert [(s.x, s.y, s.width, s.height) for s in segs] == [(0, 0, 64, 8)]

    def test_diff_with_fit_can_accept_compressed_large_region(self):
        w, h = 64, 64
        old = bytes(w * h)
        new = bytes([1] * (w * h))

        def compressed_wire_fits(seg: Segment) -> bool:
            raw_wire = bytes([0xFF]) * ((seg.pixel_count + 7) // 8)
            return len(raw_wire) <= 64 or len(zlib.compress(raw_wire, level=6)) <= 64

        segs = RecursiveBoundingBoxStrategy(min_region_pixels=1).diff_with_fit(
            old, new, w, h, 1, compressed_wire_fits
        )

        assert [(s.x, s.y, s.width, s.height) for s in segs] == [(0, 0, 64, 64)]

    def test_respects_chunk_size_budget(self):
        # Confirm segments respect both unencrypted and encrypted MTU minus header.
        w, h = 32, 32
        old = bytes(w * h)
        new = bytes([1] * (w * h))
        for chunk in (CHUNK_SIZE, ENCRYPTED_CHUNK_SIZE):
            budget = chunk - SEGMENT_HEADER_SIZE
            segs = RecursiveBoundingBoxStrategy(min_region_pixels=1).diff(
                old, new, w, h, 1, max_segment_bytes=budget
            )
            for s in segs:
                # tiny regions allowed to exceed via min_region_pixels guard,
                # but here min_region_pixels=1 so they must all fit.
                assert s.width * s.height <= budget


class TestFullImageStrategy:
    def test_always_empty(self):
        old = bytes(100)
        new = bytes([1] * 100)
        assert FullImageStrategy().diff(old, new, 10, 10, 1, 4096) == []


class TestPackSegmentsIntoPackets:
    @staticmethod
    def _seg(x, y, w, h, n):
        return Segment(x=x, y=y, width=w, height=h, pixels=bytes(n), plane=0), bytes([0xAA] * n)

    def test_empty(self):
        assert pack_segments_into_packets([], mtu=230) == []

    def test_each_packet_within_mtu(self):
        pairs = [self._seg(0, 0, 10, 10, 100) for _ in range(5)]
        pairs += [self._seg(0, 0, 5, 5, 25) for _ in range(10)]
        packets = pack_segments_into_packets(pairs, mtu=230)
        for p in packets:
            assert len(p) <= 230
            assert p[:2] == b"\x00\x77"

    def test_every_segment_appears_once(self):
        # Use unique pixel sentinels so we can detect duplicates / drops
        pairs = []
        for i in range(20):
            seg = Segment(x=i, y=0, width=4, height=4, pixels=b"", plane=0)
            wire = bytes([i] * 30)
            pairs.append((seg, wire))
        packets = pack_segments_into_packets(pairs, mtu=230)
        # Sum of all bytes after 0x0077 prefix == sum of all wire+header bytes
        total_payload = sum(len(p) - 2 for p in packets)
        # 9-byte header per segment + 30 byte payload = 39 bytes per segment, 20 segments
        assert total_payload == 20 * (SEGMENT_HEADER_SIZE + 30)

    def test_segment_flags_include_plane_and_compression(self):
        compressed = zlib.compress(bytes([0x00]) * 64)
        packets = pack_segments_into_packets(
            [(Segment(x=0, y=0, width=64, height=8, pixels=b"", plane=1, compressed=True), compressed)],
            mtu=230,
        )

        assert len(packets) == 1
        assert packets[0][:2] == b"\x00\x77"
        assert packets[0][10] == SEGMENT_FLAG_PLANE_1 | SEGMENT_FLAG_COMPRESSED
        assert zlib.decompress(packets[0][11:]) == bytes([0x00]) * 64


class TestNewBuilders:
    def test_partial_start(self):
        cmd = build_direct_write_partial_start(0xDEADBEEF)
        assert cmd == b"\x00\x76\x01\xde\xad\xbe\xef"

    def test_end_with_etag(self):
        cmd = build_direct_write_end_with_etag(refresh_mode=0, new_etag=0x01020304)
        assert cmd == b"\x00\x72\x00\x01\x02\x03\x04"

    def test_partial_data_packet(self):
        cmd = build_partial_data_packet(b"\x01\x02\x03")
        assert cmd == b"\x00\x77\x01\x02\x03"
