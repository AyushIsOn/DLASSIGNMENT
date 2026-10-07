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
        case .invalidBackendURL: "The server URL is not valid. Tap the gear icon to set it."
        case .insecureBackendURL: "Use https:// for internet servers (http:// only works for a local IP address)."
        case .invalidRequest(let message), .rateLimited(let message),
             .upstreamFailure(let message), .serviceUnavailable(let message): message
        case .server(_, _, let message): message
        case .timeout: "The request timed out. The model may still be loading - please try again."
        case .cancelled: "The request was cancelled."
        case .malformedResponse: "The server returned an unreadable response."
        case .transport(let message): message
        }
    }
}

/// Where the app finds the AcharyaGPT server. The URL is set in the in-app settings
/// (gear icon) and stored in UserDefaults; an optional ACHARYA_BACKEND_URL Info.plist
/// value and finally the local default are used as fallbacks.
enum BackendSettings {
    static let storageKey = "backendURL"
    static let defaultURL = "http://127.0.0.1:8000"

    static func currentURLString(defaults: UserDefaults = .standard, bundle: Bundle = .main) -> String {
        if let saved = defaults.string(forKey: storageKey), !saved.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return normalized(saved)
        }
        if let configured = bundle.object(forInfoDictionaryKey: "ACHARYA_BACKEND_URL") as? String,
           !configured.isEmpty {
            return normalized(configured)
        }
        return defaultURL
    }

    static func currentURL(defaults: UserDefaults = .standard) throws -> URL {
        guard let url = URL(string: currentURLString(defaults: defaults)) else {
            throw ChatAPIError.invalidBackendURL
        }
        return url
    }

    /// Trims whitespace and trailing slashes and adds a scheme when it was left out.
    static func normalized(_ raw: String) -> String {
        var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        while value.hasSuffix("/") { value.removeLast() }
        if value.isEmpty { return defaultURL }
        if !value.lowercased().hasPrefix("http://") && !value.lowercased().hasPrefix("https://") {
            let host = value.split(separator: ":").first.map(String.init) ?? value
            value = (isLocalHost(host) ? "http://" : "https://") + value
        }
        return value
    }

    /// Hosts that App Transport Security allows over plain http.
    static func isLocalHost(_ rawHost: String) -> Bool {
        let host = rawHost.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        if ["localhost", "127.0.0.1", "::1"].contains(host) || host.hasSuffix(".local") {
            return true
        }
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        return parts.count == 4 && parts.allSatisfy { part in
            guard let number = Int(part), String(number) == part else { return false }
            return (0...255).contains(number)
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

    /// Uses the server URL from the in-app settings.
    init(session: URLSession = .shared) throws {
        try self.init(session: session, baseURL: BackendSettings.currentURL())
    }

    func send(_ request: ChatRequest) async throws -> ChatResponse {
        let endpoint = baseURL.appendingPathComponent("v1/chat")
        // An 8B model answers in a few seconds, but the first request may include model loading.
        var urlRequest = URLRequest(url: endpoint, timeoutInterval: 180)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try encoder.encode(request)
        let data = try await perform(urlRequest)
        guard let decoded = try? decoder.decode(ChatResponse.self, from: data) else {
            throw ChatAPIError.malformedResponse
        }
        return decoded
    }

    func health() async throws -> HealthStatus {
        let urlRequest = URLRequest(url: baseURL.appendingPathComponent("health/ready"),
                                    timeoutInterval: 20)
        do {
            let data = try await perform(urlRequest)
            guard let decoded = try? decoder.decode(HealthStatus.self, from: data) else {
                throw ChatAPIError.malformedResponse
            }
            return decoded
        } catch ChatAPIError.serviceUnavailable(let message) {
            return HealthStatus(status: "not_ready", ready: false, model: nil, detail: message)
        }
    }

    private func perform(_ urlRequest: URLRequest) async throws -> Data {
        do {
            let (data, response) = try await session.data(for: urlRequest)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ChatAPIError.malformedResponse
            }
            guard (200...299).contains(httpResponse.statusCode) else {
                throw normalizedError(status: httpResponse.statusCode, data: data)
            }
            return data
        } catch let error as ChatAPIError {
            throw error
        } catch let error as URLError where error.code == .timedOut {
            throw ChatAPIError.timeout
        } catch let error as URLError where error.code == .cancelled {
            throw ChatAPIError.cancelled
        } catch is CancellationError {
            throw ChatAPIError.cancelled
        } catch let error as URLError where error.code == .cannotConnectToHost || error.code == .cannotFindHost {
            throw ChatAPIError.transport("Cannot reach the server at \(baseURL.absoluteString). Is it running?")
        } catch {
            throw ChatAPIError.transport(error.localizedDescription)
        }
    }

    private func normalizedError(status: Int, data: Data) -> ChatAPIError {
        let body = try? decoder.decode(APIErrorEnvelope.self, from: data).error
        let code = body?.code ?? "http_\(status)"
        let message = body?.message ?? "The server returned HTTP \(status)."
        return switch status {
        case 422: .invalidRequest(message)
        case 429: .rateLimited(message)
        case 502: .upstreamFailure(message)
        case 503: .serviceUnavailable(message)
        default: .server(status: status, code: code, message: message)
        }
    }

    private static func validate(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(), let host = url.host, !host.isEmpty else {
            throw ChatAPIError.invalidBackendURL
        }
        if scheme == "https" { return }
        if scheme == "http" && BackendSettings.isLocalHost(host) { return }
        if scheme == "http" { throw ChatAPIError.insecureBackendURL }
        throw ChatAPIError.invalidBackendURL
    }
}
