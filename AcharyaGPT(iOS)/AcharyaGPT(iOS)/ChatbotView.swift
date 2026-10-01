import SwiftUI

struct ChatbotView: View {
    @State private var hasScrolled = false
    @StateObject private var chatViewModel: ChatViewModel
    @FocusState private var isFocused: Bool

    init(viewModel: ChatViewModel? = nil) {
        _chatViewModel = StateObject(wrappedValue: viewModel ?? ChatViewModel())
    }

    var body: some View {
        ZStack {
            ScrollViewReader { proxy in
                ScrollView(showsIndicators: false) {
                    LazyVStack(spacing: 16) {
                        scrollDetection
                        Image("acharyaLogoLarge")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 60, height: 51)
                            .padding()
                        IntroductionView()
                        ForEach(chatViewModel.chatMessages) { message in
                            messageView(message)
                        }
                        Color.clear.frame(height: 130).id("bottom")
                    }
                }
                .onChange(of: chatViewModel.chatMessages) { _, messages in
                    guard !messages.isEmpty else { return }
                    withAnimation { proxy.scrollTo("bottom") }
                }
                .coordinateSpace(name: "scroll")
                .safeAreaInset(edge: .top) { Color.clear.frame(height: 70) }
                .overlay(AcharyaNavigationView(hasScrolled: $hasScrolled))
            }
            inputPanel.frame(maxWidth: .infinity)
        }
        .background(Color(red: 0.05, green: 0.06, blue: 0.06))
        .preferredColorScheme(.dark)
    }

    private func messageView(_ message: ChatMessage) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(message.role == .user ? "senderProfile" : "acharyaLogo")
                    .resizable().scaledToFit().frame(width: 16, height: 16)
                Text(message.role == .user ? "You" : "AcharyaGPT").font(.caption.bold())
            }
            Text(message.text)
                .textSelection(.enabled)
                .font(.system(size: 16))
                .foregroundStyle(Color(red: 0.92, green: 0.92, blue: 0.92))
                .frame(maxWidth: .infinity, alignment: .leading)
            if let response = message.response {
                HStack {
                    StatusBadge(text: response.outcome, color: response.outcome == "answered" ? .green : .orange)
                    StatusBadge(text: response.mode, color: .blue)
                    if response.outcome == "urgent" || response.outcome == "refused" {
                        StatusBadge(text: "safety", color: .red)
                    }
                }
                if let warning = response.warning {
                    StatusBadge(text: warning, color: .yellow)
                }
                ForEach(response.citations) { CitationView(citation: $0) }
            }
        }
        .padding(14)
        .background(
            LinearGradient(colors: [.white.opacity(0.08), .white.opacity(0.18)], startPoint: .topLeading, endPoint: .bottomTrailing)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.3)))
        .padding(.horizontal)
    }

    private var scrollDetection: some View {
        GeometryReader { proxy in
            Color.clear.preference(key: ScrollPreferenceKey.self, value: proxy.frame(in: .named("scroll")).minY)
        }
        .frame(height: 0)
        .onPreferenceChange(ScrollPreferenceKey.self) { value in
            withAnimation(.easeInOut(duration: 0.4)) { hasScrolled = value < 0 }
        }
    }

    private var inputPanel: some View {
        VStack(spacing: 10) {
            if let error = chatViewModel.errorMessage {
                HStack {
                    Text(error).font(.caption).foregroundStyle(.red)
                    Spacer()
                    Button("Retry") { chatViewModel.retry() }
                }
                .padding(.horizontal)
            }
            HStack(alignment: .bottom) {
                TextField("What's your query?", text: $chatViewModel.message, axis: .vertical)
                    .focused($isFocused)
                    .lineLimit(1...6)
                    .padding(16)
                    .background(Color.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.4)))
                if chatViewModel.isWaitingForResponse {
                    Button("Cancel") { chatViewModel.cancel() }
                        .frame(height: 52)
                } else {
                    Button { chatViewModel.sendMessage() } label: {
                        Image("sendButton").frame(width: 52, height: 52)
                    }
                    .disabled(!chatViewModel.canSend)
                    .opacity(chatViewModel.canSend ? 1 : 0.5)
                }
            }
            .padding(.horizontal)
        }
        .padding(.top, 12)
        .padding(.bottom, 28)
        .background(.ultraThinMaterial)
        .frame(maxHeight: .infinity, alignment: .bottom)
        .ignoresSafeArea()
    }
}

#Preview {
    ChatbotView()
}
