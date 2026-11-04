// Draws the two images scripts/icons.sh makes the app icons from:
// macos.png, the logo on a white rounded tile laid out on Apple's 1024 px
// icon grid (an 824 px tile with a soft shadow), and windows.png, the logo
// alone, filling the square.
//
//   swift scripts/icons.swift <logo.png> <out dir>
import CoreGraphics
import Foundation
import ImageIO

func canvas(_ w: Int, _ h: Int, _ data: UnsafeMutableRawPointer? = nil) -> CGContext {
    CGContext(data: data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
              space: CGColorSpace(name: CGColorSpace.sRGB)!,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
}

/// The drawn (not transparent) part of the image, in Core Graphics
/// coordinates (origin bottom left), so the drawing is centred whatever
/// margin the file has.
func drawnBounds(_ img: CGImage) -> CGRect {
    let w = img.width, h = img.height
    var px = [UInt8](repeating: 0, count: w * h * 4)
    px.withUnsafeMutableBytes { canvas(w, h, $0.baseAddress).draw(img, in: CGRect(x: 0, y: 0, width: w, height: h)) }
    var minX = w, maxX = -1, minRow = h, maxRow = -1
    for row in 0..<h {
        for x in 0..<w where px[(row * w + x) * 4 + 3] > 8 {
            minX = min(minX, x); maxX = max(maxX, x)
            minRow = min(minRow, row); maxRow = max(maxRow, row)
        }
    }
    // row 0 of the buffer is the top of the image
    return CGRect(x: minX, y: h - 1 - maxRow, width: maxX - minX + 1, height: maxRow - minRow + 1)
}

/// Draws `img` so that its drawn part `drawn` fits `box`, centred, in proportion.
func draw(_ img: CGImage, drawn: CGRect, in ctx: CGContext, fitting box: CGRect) {
    let s = min(box.width / drawn.width, box.height / drawn.height)
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: box.midX - drawn.midX * s, y: box.midY - drawn.midY * s,
                             width: CGFloat(img.width) * s, height: CGFloat(img.height) * s))
}

func save(_ ctx: CGContext, _ url: URL) {
    let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
    guard CGImageDestinationFinalize(dest) else { fatalError("cannot write \(url.path)") }
}

let args = CommandLine.arguments
guard args.count == 3,
      let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[1]) as CFURL, nil),
      let logo = CGImageSourceCreateImageAtIndex(source, 0, nil)
else {
    FileHandle.standardError.write("usage: swift scripts/icons.swift <logo.png> <out dir>\n".data(using: .utf8)!)
    exit(2)
}
let out = URL(fileURLWithPath: args[2])
let drawn = drawnBounds(logo)

// macOS: tile 824 px with corner radius 185, the 100 px margin holds the shadow
let mac = canvas(1024, 1024)
let tile = CGPath(roundedRect: CGRect(x: 100, y: 100, width: 824, height: 824), cornerWidth: 185, cornerHeight: 185, transform: nil)
mac.saveGState()
mac.setShadow(offset: CGSize(width: 0, height: -10), blur: 20, color: CGColor(gray: 0, alpha: 0.3))
mac.setFillColor(CGColor(gray: 1, alpha: 1))
mac.addPath(tile)
mac.fillPath()
mac.restoreGState()
mac.addPath(tile)
mac.clip()
draw(logo, drawn: drawn, in: mac, fitting: CGRect(x: 167, y: 167, width: 690, height: 690))
save(mac, out.appendingPathComponent("macos.png"))

// Windows: the logo alone, nearly to the edge
let win = canvas(1024, 1024)
draw(logo, drawn: drawn, in: win, fitting: CGRect(x: 24, y: 24, width: 976, height: 976))
save(win, out.appendingPathComponent("windows.png"))
