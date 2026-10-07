import SwiftUI

struct ScrollPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = nextValue() }
}

struct AcharyaNavigationView: View {
    @Binding var hasScrolled: Bool
    var onNewChat: () -> Void = {}
    var onSettings: () -> Void = {}

    var body: some View {
        HStack(spacing: 16) {
            Text("AcharyaGPT").font(.headline).foregroundStyle(.white)
            Spacer()
            Button(action: onNewChat) {
                Image(systemName: "square.and.pencil")
            }
            .accessibilityLabel("New chat")
            Button(action: onSettings) {
                Image(systemName: "gearshape")
            }
            .accessibilityLabel("Server settings")
        }
        .font(.title3)
        .foregroundStyle(.white.opacity(0.9))
        .padding()
        .background(hasScrolled ? Color.black.opacity(0.75) : Color.clear)
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

struct IntroductionView: View {
    var body: some View {
        VStack(spacing: 8) {
            Text("Your Ayurveda study companion")
                .font(.title3.bold())
                .foregroundStyle(.white)
            Text("A Qwen3-8B model fine-tuned on Ayurvedic knowledge, with sources from its knowledge base. Educational only - not medical advice.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 24)
    }
}

struct SuggestionsView: View {
    let onSelect: (String) -> Void
    private let suggestions = [
        "What are the three doshas?",
        "What is the modern equivalent of Amlapitta?",
        "Which dosha is predominant in Ardita?",
        "What are the symptoms of Kamala?",
        "What is Panchakarma?",
        "Which Ayurvedic herbs are used for Asthma?",
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Try asking").font(.caption.bold()).foregroundStyle(.secondary)
            ForEach(suggestions, id: \.self) { suggestion in
                Button { onSelect(suggestion) } label: {
                    Text(suggestion)
                        .font(.subheadline)
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.leading)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                }
            }
        }
        .padding(.horizontal)
    }
}

struct TypingIndicator: View {
    var body: some View {
        HStack(spacing: 10) {
            Image("acharyaLogo").resizable().scaledToFit().frame(width: 16, height: 16)
            ProgressView().tint(.white)
            Text("AcharyaGPT is thinking…").font(.subheadline).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(14)
        .background(.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
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
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(citation.page > 0 ? "\(citation.source) · page \(citation.page)" : citation.source)
                .font(.caption.bold())
            Text(citation.text)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(expanded ? nil : 2)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onTapGesture { expanded.toggle() }
    }
}

/// Lets the user point the app at the server: the Mac (simulator), a Mac on the same
/// Wi-Fi (real iPhone), or the https URL printed by scripts/lightning_serve.sh.
struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var draft = BackendSettings.currentURLString()
    @State private var status: String?
    @State private var checking = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://127.0.0.1:8000", text: $draft)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Button(checking ? "Checking…" : "Test connection") { test() }
                        .disabled(checking)
                    if let status {
                        Text(status).font(.footnote).foregroundStyle(.secondary)
                    }
                } header: {
                    Text("AcharyaGPT server")
                } footer: {
                    Text("Simulator on the same Mac: http://127.0.0.1:8000\nReal iPhone: http://<your-Mac-IP>:8000 (run mac_serve.sh --lan)\nLightning GPU: the https://… URL printed by lightning_serve.sh")
                }
                Section {
                    Button("Reset to default") { draft = BackendSettings.defaultURL }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        UserDefaults.standard.set(BackendSettings.normalized(draft),
                                                  forKey: BackendSettings.storageKey)
                        dismiss()
                    }
                }
            }
        }
    }

    private func test() {
        let value = BackendSettings.normalized(draft)
        draft = value
        checking = true
        status = nil
        Task { @MainActor in
            defer { checking = false }
            guard let url = URL(string: value) else {
                status = ChatAPIError.invalidBackendURL.localizedDescription
                return
            }
            do {
                let health = try await ChatAPIClient(baseURL: url).health()
                status = health.ready
                    ? "Connected ✓  model: \(health.model ?? "unknown")"
                    : "Server reachable, but the model is not ready: \(health.detail ?? "loading")"
            } catch {
                status = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            }
        }
    }
}
