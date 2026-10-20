"""Render icon.svg to transparent sRGB PNGs: render_icon.py OUTPUT SIZE [OUTPUT SIZE ...]."""

import sys
from pathlib import Path

from AppKit import (
    NSBitmapImageRep,
    NSColorSpace,
    NSCompositingOperationCopy,
    NSGraphicsContext,
    NSImage,
    NSMakeRect,
    NSPNGFileType,
)

SOURCE = Path(__file__).with_name("icon.svg")


def render(output: str, size: int) -> None:
    image = NSImage.alloc().initWithContentsOfFile_(str(SOURCE))
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(  # noqa: E501 - one AppKit selector
        None, size, size, 8, 4, True, False, "NSDeviceRGBColorSpace", 0, 0
    )
    # Tag as sRGB so the colours match the website's hex values.
    rep = rep.bitmapImageRepByRetaggingWithColorSpace_(NSColorSpace.sRGBColorSpace())
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    image.drawInRect_fromRect_operation_fraction_(
        NSMakeRect(0, 0, size, size), NSMakeRect(0, 0, 0, 0), NSCompositingOperationCopy, 1.0
    )
    NSGraphicsContext.restoreGraphicsState()
    data = rep.representationUsingType_properties_(NSPNGFileType, {})
    if not data.writeToFile_atomically_(output, True):
        raise SystemExit(f"Cannot write {output}")


if __name__ == "__main__":
    pairs = sys.argv[1:]
    if not pairs or len(pairs) % 2:
        raise SystemExit(__doc__)
    for output, size in zip(pairs[::2], pairs[1::2], strict=True):
        render(output, int(size))
