import SwiftUI

// Values correspond to the approved UI prototype root and light/dark theme tokens.
enum Tokens {
    static let none: CGFloat = 0, hairline: CGFloat = 1, quoteRule: CGFloat = 2, ringLine: CGFloat = 3, labelWidth: CGFloat = 52
    static let xs: CGFloat = 11, sm: CGFloat = 12, md: CGFloat = 13.5, base: CGFloat = 15, lg: CGFloat = 17, xl: CGFloat = 22, xxl: CGFloat = 28
    static let sp1: CGFloat = 4, sp2: CGFloat = 8, sp3: CGFloat = 12, sp4: CGFloat = 16, sp5: CGFloat = 20, sp6: CGFloat = 24, sp8: CGFloat = 32
    static let small: CGFloat = 6, medium: CGFloat = 10, large: CGFloat = 14, pill: CGFloat = 999
    static let dot: CGFloat = 7, dotGap: CGFloat = 5, dotLift: CGFloat = 5, dotRow: CGFloat = 24
    static let duration = 0.14
    private static func color(_ light: UInt32, _ dark: UInt32) -> Color {
        Color(uiColor: UIColor { traits in
            let value = traits.userInterfaceStyle == .dark ? dark : light
            return UIColor(red: Double((value >> 16) & 255) / 255, green: Double((value >> 8) & 255) / 255, blue: Double(value & 255) / 255, alpha: 1)
        })
    }
    static let background = color(0xf4f4f2, 0x121315), sidebar = color(0xececea, 0x17181b)
    static let surface = color(0xffffff, 0x1c1d21), surface2 = color(0xf7f7f5, 0x212226), surface3 = color(0xefefec, 0x2a2b30)
    static let text = color(0x1b1c1e, 0xececee), text2 = color(0x55585e, 0xa8abb2), text3 = color(0x8b8e94, 0x75787f)
    static let border = color(0xe3e3df, 0x2c2d32), borderStrong = color(0xd2d2cd, 0x3a3b41)
    static let accent = color(0x2f6feb, 0x6b9bff), accentHover = color(0x2a62d1, 0x87adff), accentSoft = color(0xe8effd, 0x1f2a42), accentInk = color(0xffffff, 0x0f1115)
    static let ok = color(0x16834f, 0x4fcf8e), okSoft = color(0xe4f3ea, 0x173225), warn = color(0xa76400, 0xeab155), warnSoft = color(0xfbf0dc, 0x3a2c14), error = color(0xc4352f, 0xff8078), errorSoft = color(0xfbe8e6, 0x3c1d1b)
    static let segments = [color(0x2f6feb, 0x6b9bff), color(0x7b61ff, 0xa08cff), color(0x16a3a3, 0x3cc6c6), color(0xe8933a, 0xf0a75b), color(0xb4b7bd, 0x5d6067), color(0xd9b44a, 0xc9a640)]
}
struct AppCard: ViewModifier {
    func body(content: Content) -> some View { content.padding(Tokens.sp4).frame(maxWidth: .infinity, alignment: .leading).background(Tokens.surface, in: RoundedRectangle(cornerRadius: Tokens.large)).overlay { RoundedRectangle(cornerRadius: Tokens.large).stroke(Tokens.border) } }
}
extension View { func appCard() -> some View { modifier(AppCard()) } }
