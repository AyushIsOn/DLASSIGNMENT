import SwiftUI

struct ScrollPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

struct AcharyaNavigationView: View {
    @Binding var hasScrolled: Bool

    var body: some View {
        HStack {
            Text("AcharyaGPT").font(.headline).foregroundStyle(.white)
            Spacer()
        }
        .padding()
        .background(hasScrolled ? Color.black.opacity(0.75) : Color.clear)
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

struct IntroductionView: View {
    var body: some View {
        VStack(spacing: 8) {
            Text("Ayurvedic knowledge, grounded in sources")
                .font(.title3.bold())
                .foregroundStyle(.white)
            Text("Ask a question and review the citations. This app is informational, not medical advice.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 24)
    }
}

struct StatusBadge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text.replacingOccurrences(of: "_", with: " ").uppercased())
            .font(.caption2.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .foregroundStyle(color)
            .background(color.opacity(0.15), in: Capsule())
    }
}

struct CitationView: View {
    let citation: Citation

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(citation.source) · page \(citation.page)").font(.caption.bold())
            Text(citation.text).font(.caption).foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
    }
}
