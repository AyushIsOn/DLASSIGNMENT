import SwiftUI

struct ChatbotView: View {
    @State private var hasScrolled = false
    @State private var showSettings = false
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
                        if chatViewModel.chatMessages.isEmpty && chatViewModel.pendingMessage == nil {
                            SuggestionsView { suggestion in
                                isFocused = false
                                chatViewModel.message = suggestion
                                chatViewModel.sendMessage()
                            }
                        }
                        ForEach(chatViewModel.chatMessages) { message in
                            messageView(message)
                        }
                        if let pending = chatViewModel.pendingMessage {
                            messageView(ChatMessage(role: .user, text: pending))
                            TypingIndicator()
                        }
                        Color.clear.frame(height: 130).id("bottom")
                    }
                }
                .scrollDismissesKeyboard(.immediately)
                // tapping anywhere in the conversation closes the keyboard
                .simultaneousGesture(TapGesture().onEnded { isFocused = false })
                .onChange(of: chatViewModel.chatMessages) { _, messages in
                    guard !messages.isEmpty else { return }
                    withAnimation { proxy.scrollTo("bottom") }
                }
                .onChange(of: chatViewModel.pendingMessage) { _, pending in
                    guard pending != nil else { return }
                    withAnimation { proxy.scrollTo("bottom") }
                }
                .coordinateSpace(name: "scroll")
                .safeAreaInset(edge: .top) { Color.clear.frame(height: 70) }
                .overlay(
                    AcharyaNavigationView(
                        hasScrolled: $hasScrolled,
                        onNewChat: { chatViewModel.clearConversation() },
                        onSettings: { showSettings = true }
                    )
                )
            }
            inputPanel.frame(maxWidth: .infinity)
        }
        .background(Color(red: 0.05, green: 0.06, blue: 0.06))
        .preferredColorScheme(.dark)
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
    }

    private func modeLabel(_ mode: String) -> String {
        switch mode {
        case "finetuned_rag": "Fine-tuned + RAG"
        case "finetuned": "Fine-tuned"
        case "safety": "Safety"
        default: mode
        }
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
                    StatusBadge(text: modeLabel(response.mode),
                                color: response.mode == "safety" ? .red : .teal)
                    if response.outcome == "urgent" {
                        StatusBadge(text: "urgent", color: .red)
                    }
                    if let latency = response.latencyMs, response.mode != "safety" {
                        Text(String(format: "%.1f s", Double(latency) / 1000))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                if let warning = response.warning {
                    StatusBadge(text: warning, color: .yellow)
                }
                if !response.citations.isEmpty {
                    Text("Sources").font(.caption.bold()).foregroundStyle(.secondary)
                    ForEach(response.citations) { CitationView(citation: $0) }
                }
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
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
                .padding(.horizontal)
            }
            HStack(alignment: .bottom) {
                TextField("What's your query?", text: $chatViewModel.message, axis: .vertical)
                    .focused($isFocused)
                    .toolbar {
                        ToolbarItemGroup(placement: .keyboard) {
                            Spacer()
                            Button("Done") { isFocused = false }
                        }
                    }
                    .lineLimit(1...6)
                    .padding(16)
                    .background(Color.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 14))
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(.white.opacity(0.4)))
                if chatViewModel.isWaitingForResponse {
                    Button("Cancel") { chatViewModel.cancel() }
                        .frame(height: 52)
                } else {
                    Button {
                        isFocused = false
                        chatViewModel.sendMessage()
                    } label: {
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
