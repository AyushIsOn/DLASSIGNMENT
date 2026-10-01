import Foundation

enum ChatRole: String, Codable {
    case user
    case assistant
}

struct ChatHistoryMessage: Codable, Equatable {
    let role: ChatRole
    let content: String
}

struct ChatRequest: Codable, Equatable {
    let message: String
    let history: [ChatHistoryMessage]
}

struct Citation: Codable, Equatable, Identifiable {
    let chunkId: String
    let source: String
    let page: Int
    let text: String

    var id: String { chunkId }

    enum CodingKeys: String, CodingKey {
        case chunkId = "chunk_id"
        case source
        case page
        case text
    }
}

struct RetrievalScores: Codable, Equatable {
    let bm25: Double
    let dense: Double?
    let rerank: Double?
}

struct ChatResponse: Codable, Equatable {
    let answer: String
    let citations: [Citation]
    let scores: [RetrievalScores]
    let mode: String
    let warning: String?
    let outcome: String
}

struct APIErrorEnvelope: Codable, Equatable {
    let error: APIErrorBody
}

struct APIErrorBody: Codable, Equatable {
    let code: String
    let message: String
}

struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    let role: ChatRole
    let text: String
    let response: ChatResponse?

    init(id: UUID = UUID(), role: ChatRole, text: String, response: ChatResponse? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.response = response
    }
}
