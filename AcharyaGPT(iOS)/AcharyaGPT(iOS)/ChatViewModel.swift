import Foundation

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var message = ""
    @Published private(set) var chatMessages: [ChatMessage] = []
    @Published private(set) var isWaitingForResponse = false
    @Published private(set) var errorMessage: String?

    private let client: ChatAPIClientProtocol
    private var requestTask: Task<Void, Never>?
    private var retryMessage: String?
    private var lastSubmittedMessage: String?

    init(client: ChatAPIClientProtocol? = nil) {
        if let client {
            self.client = client
        } else {
            self.client = (try? ChatAPIClient()) ?? UnavailableChatAPIClient()
        }
    }

    var canSend: Bool {
        !isWaitingForResponse && !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func sendMessage() {
        submit(message)
    }

    func retry() {
        guard let retryMessage else { return }
        submit(retryMessage, isRetry: true)
    }

    func cancel() {
        requestTask?.cancel()
    }

    private func submit(_ rawMessage: String, isRetry: Bool = false) {
        let trimmed = rawMessage.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isWaitingForResponse else { return }
        guard isRetry || trimmed.caseInsensitiveCompare(lastSubmittedMessage ?? "") != .orderedSame else {
            errorMessage = "That message was already sent."
            return
        }

        let history = completeHistoryPairs()
        isWaitingForResponse = true
        errorMessage = nil
        retryMessage = nil
        lastSubmittedMessage = trimmed

        requestTask = Task { [weak self] in
            guard let self else { return }
            do {
                let response = try await client.send(ChatRequest(message: trimmed, history: history))
                guard !Task.isCancelled else { throw ChatAPIError.cancelled }
                chatMessages.append(ChatMessage(role: .user, text: trimmed))
                chatMessages.append(ChatMessage(role: .assistant, text: response.answer, response: response))
                message = ""
            } catch let error as ChatAPIError {
                errorMessage = error.localizedDescription
                retryMessage = error == .cancelled ? nil : trimmed
            } catch is CancellationError {
                errorMessage = ChatAPIError.cancelled.localizedDescription
                retryMessage = nil
            } catch {
                errorMessage = error.localizedDescription
                retryMessage = trimmed
            }
            isWaitingForResponse = false
            requestTask = nil
        }
    }

    private func completeHistoryPairs() -> [ChatHistoryMessage] {
        var history: [ChatHistoryMessage] = []
        var index = 0
        while index + 1 < chatMessages.count {
            let user = chatMessages[index]
            let assistant = chatMessages[index + 1]
            guard user.role == .user, assistant.role == .assistant else { break }
            history.append(ChatHistoryMessage(role: .user, content: user.text))
            history.append(ChatHistoryMessage(role: .assistant, content: assistant.text))
            index += 2
        }
        return Array(history.suffix(20))
    }
}

private struct UnavailableChatAPIClient: ChatAPIClientProtocol {
    func send(_ request: ChatRequest) async throws -> ChatResponse {
        throw ChatAPIError.invalidBackendURL
    }
}
