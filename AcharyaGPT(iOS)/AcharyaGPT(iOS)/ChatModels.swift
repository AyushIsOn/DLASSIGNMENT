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
    let model: String?
    let latencyMs: Int?

    enum CodingKeys: String, CodingKey {
        case answer, citations, scores, mode, warning, outcome, model
        case latencyMs = "latency_ms"
    }

    init(answer: String, citations: [Citation], scores: [RetrievalScores], mode: String,
         warning: String?, outcome: String, model: String? = nil, latencyMs: Int? = nil) {
        self.answer = answer
        self.citations = citations
        self.scores = scores
        self.mode = mode
        self.warning = warning
        self.outcome = outcome
        self.model = model
        self.latencyMs = latencyMs
    }
}

struct HealthStatus: Codable, Equatable {
    let status: String
    let ready: Bool
    let model: String?
    let detail: String?
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
