#!/usr/bin/env python
"""Generate the VIVE Labeler application icon (ICO + PNG)."""

import struct
import zlib
from pathlib import Path


def create_png(size: int) -> bytes:
    """Create a PNG icon of the given size using raw bytes (no Pillow needed)."""
    # Draw a simple but recognizable icon:
    # - Dark rounded background (#1e1e2e)
    # - Green play/label triangle (#40a02b)
    # - White "V" letter for VIVE

    pixels = []
    cx, cy = size / 2, size / 2
    r = size / 2 - 1

    for y in range(size):
        row = []
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = (dx * dx + dy * dy) ** 0.5

            # Rounded square mask (superellipse)
            corner_r = size * 0.2
            in_shape = True
            if x < corner_r and y < corner_r:
                in_shape = ((corner_r - x) ** 2 + (corner_r - y) ** 2) <= corner_r ** 2
            elif x >= size - corner_r and y < corner_r:
                in_shape = ((x - (size - corner_r)) ** 2 + (corner_r - y) ** 2) <= corner_r ** 2
            elif x < corner_r and y >= size - corner_r:
                in_shape = ((corner_r - x) ** 2 + (y - (size - corner_r)) ** 2) <= corner_r ** 2
            elif x >= size - corner_r and y >= size - corner_r:
                in_shape = ((x - (size - corner_r)) ** 2 + (y - (size - corner_r)) ** 2) <= corner_r ** 2

            if not in_shape:
                row.extend([0, 0, 0, 0])  # transparent
                continue

            # Background: dark (#1e1e2e)
            pr, pg, pb, pa = 0x1e, 0x1e, 0x2e, 255

            # Draw "V" letter in white
            nx, ny = x / size, y / size  # normalized coords

            # Left stroke of V: from (0.2, 0.2) to (0.5, 0.8)
            if 0.18 <= ny <= 0.82:
                t = (ny - 0.2) / 0.6
                vx_left = 0.2 + t * 0.3
                stroke_w = 0.07
                if abs(nx - vx_left) < stroke_w:
                    pr, pg, pb = 255, 255, 255

            # Right stroke of V: from (0.8, 0.2) to (0.5, 0.8)
            if 0.18 <= ny <= 0.82:
                t = (ny - 0.2) / 0.6
                vx_right = 0.8 - t * 0.3
                stroke_w = 0.07
                if abs(nx - vx_right) < stroke_w:
                    pr, pg, pb = 255, 255, 255

            # Green accent dot bottom-right
            dot_cx, dot_cy = 0.78 * size, 0.78 * size
            dot_r = size * 0.1
            dot_dist = ((x - dot_cx) ** 2 + (y - dot_cy) ** 2) ** 0.5
            if dot_dist < dot_r:
                pr, pg, pb = 0x40, 0xa0, 0x2b

            row.extend([pr, pg, pb, pa])
        pixels.append(bytes(row))

    # Encode as PNG
    def make_png(width, height, rows):
        def chunk(chunk_type, data):
            c = chunk_type + data
            crc = struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack('>I', len(data)) + c + crc

        sig = b'\x89PNG\r\n\x1a\n'
        ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
        raw_data = b''
        for row in rows:
            raw_data += b'\x00' + row  # filter byte + row data
        compressed = zlib.compress(raw_data)
        return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')

    return make_png(size, size, pixels)


def create_ico(png_data_list: list[tuple[int, bytes]]) -> bytes:
    """Create an ICO file from multiple PNG images."""
    num = len(png_data_list)
    # ICO header: reserved(2) + type(2) + count(2) = 6 bytes
    header = struct.pack('<HHH', 0, 1, num)

    # Calculate offsets
    dir_entry_size = 16
    offset = 6 + num * dir_entry_size

    entries = b''
    image_data = b''
    for size, png_bytes in png_data_list:
        w = size if size < 256 else 0
        h = size if size < 256 else 0
        entry = struct.pack('<BBBBHHII',
                            w, h,       # width, height (0 = 256)
                            0,          # color palette
                            0,          # reserved
                            1,          # color planes
                            32,         # bits per pixel
                            len(png_bytes),  # size
                            offset)     # offset
        entries += entry
        image_data += png_bytes
        offset += len(png_bytes)

    return header + entries + image_data


def main():
    assets_dir = Path(__file__).parent.parent / "assets"
    assets_dir.mkdir(exist_ok=True)

    sizes = [16, 32, 48, 64, 128, 256]
    png_list = []

    for s in sizes:
        png_data = create_png(s)
        png_list.append((s, png_data))

    # Save 256px PNG
    png_path = assets_dir / "icon.png"
    with open(png_path, 'wb') as f:
        f.write(png_list[-1][1])
    print(f"Created {png_path}")

    # Save ICO with all sizes
    ico_path = assets_dir / "icon.ico"
    ico_data = create_ico(png_list)
    with open(ico_path, 'wb') as f:
        f.write(ico_data)
    print(f"Created {ico_path}")


if __name__ == "__main__":
    main()
