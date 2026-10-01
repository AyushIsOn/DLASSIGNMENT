import Foundation

protocol ChatAPIClientProtocol {
    func send(_ request: ChatRequest) async throws -> ChatResponse
}

enum ChatAPIError: Error, Equatable, LocalizedError {
    case invalidBackendURL
    case insecureBackendURL
    case invalidRequest(String)
    case rateLimited(String)
    case upstreamFailure(String)
    case serviceUnavailable(String)
    case server(status: Int, code: String, message: String)
    case timeout
    case cancelled
    case malformedResponse
    case transport(String)

    var errorDescription: String? {
        switch self {
        case .invalidBackendURL: "The backend URL is not configured."
        case .insecureBackendURL: "The backend URL must use HTTPS."
        case .invalidRequest(let message), .rateLimited(let message),
             .upstreamFailure(let message), .serviceUnavailable(let message): message
        case .server(_, _, let message): message
        case .timeout: "The request timed out. Please try again."
        case .cancelled: "The request was cancelled."
        case .malformedResponse: "The server returned an unreadable response."
        case .transport(let message): message
        }
    }
}

struct ChatAPIClient: ChatAPIClientProtocol {
    private let session: URLSession
    private let baseURL: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(session: URLSession = .shared, baseURL: URL) throws {
        try Self.validate(baseURL)
        self.session = session
        self.baseURL = baseURL
    }

    init(session: URLSession = .shared, bundle: Bundle = .main) throws {
        guard let value = bundle.object(forInfoDictionaryKey: "ACHARYA_BACKEND_URL") as? String,
              let url = URL(string: value) else {
            throw ChatAPIError.invalidBackendURL
        }
        try self.init(session: session, baseURL: url)
    }

    func send(_ request: ChatRequest) async throws -> ChatResponse {
        let endpoint = baseURL.appendingPathComponent("v1/chat")
        var urlRequest = URLRequest(url: endpoint, timeoutInterval: 30)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try encoder.encode(request)

        do {
            let (data, response) = try await session.data(for: urlRequest)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ChatAPIError.malformedResponse
            }
            guard (200...299).contains(httpResponse.statusCode) else {
                throw normalizedError(status: httpResponse.statusCode, data: data)
            }
            guard let decoded = try? decoder.decode(ChatResponse.self, from: data) else {
                throw ChatAPIError.malformedResponse
            }
            return decoded
        } catch let error as ChatAPIError {
            throw error
        } catch let error as URLError where error.code == .timedOut {
            throw ChatAPIError.timeout
        } catch let error as URLError where error.code == .cancelled {
            throw ChatAPIError.cancelled
        } catch is CancellationError {
            throw ChatAPIError.cancelled
        } catch {
            throw ChatAPIError.transport(error.localizedDescription)
        }
    }

    private func normalizedError(status: Int, data: Data) -> ChatAPIError {
        let body = try? decoder.decode(APIErrorEnvelope.self, from: data).error
        let code = body?.code ?? "http_\(status)"
        let message = body?.message ?? "The server returned HTTP \(status)."
        switch status {
        case 422: .invalidRequest(message)
        case 429: .rateLimited(message)
        case 502: .upstreamFailure(message)
        case 503: .serviceUnavailable(message)
        default: .server(status: status, code: code, message: message)
        }
    }

    private static func validate(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(), let host = url.host else {
            throw ChatAPIError.invalidBackendURL
        }
        if scheme == "https" { return }
        #if DEBUG
        let loopbackHosts = ["localhost", "127.0.0.1", "::1"]
        if scheme == "http" && loopbackHosts.contains(host.lowercased()) { return }
        #endif
        throw ChatAPIError.insecureBackendURL
    }
}
