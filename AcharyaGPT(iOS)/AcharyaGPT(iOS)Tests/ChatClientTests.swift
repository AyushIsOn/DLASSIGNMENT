import XCTest
@testable import AcharyaGPT_iOS_

final class MockURLProtocol: URLProtocol {
    static var status = 200
    static var data = Data()
    static var error: Error?
    static var delay: TimeInterval = 0
    static var requests: [URLRequest] = []

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Self.requests.append(request)
        if let error = Self.error {
            client?.urlProtocol(self, didFailWithError: error)
            return
        }
        let finish = {
            guard let response = HTTPURLResponse(url: self.request.url!, statusCode: Self.status, httpVersion: nil, headerFields: nil) else { return }
            self.client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            self.client?.urlProtocol(self, didLoad: Self.data)
            self.client?.urlProtocolDidFinishLoading(self)
        }
        if Self.delay > 0 {
            DispatchQueue.global().asyncAfter(deadline: .now() + Self.delay, execute: finish)
        } else {
            finish()
        }
    }

    override func stopLoading() {}
}

final class ChatClientTests: XCTestCase {
    private var client: ChatAPIClient!

    override func setUpWithError() throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        client = try ChatAPIClient(session: URLSession(configuration: configuration), baseURL: URL(string: "http://127.0.0.1:8000")!)
        MockURLProtocol.status = 200
        MockURLProtocol.data = Data()
        MockURLProtocol.error = nil
        MockURLProtocol.delay = 0
        MockURLProtocol.requests = []
    }

    func testSuccessDecodesCitationsNullableScoresAndUnknownWarning() async throws {
        MockURLProtocol.data = try fixture("success")
        let response = try await client.send(ChatRequest(message: "Hello", history: []))
        XCTAssertEqual(response.citations.first?.page, 2)
        XCTAssertNil(response.scores.first?.dense)
        XCTAssertEqual(response.warning, "future_warning")
    }

    func testNormalizedHTTPFailures() async throws {
        for (status, fixtureName) in [(422, "error-422"), (429, "error-429"), (502, "error-502"), (503, "error-503")] {
            MockURLProtocol.status = status
            MockURLProtocol.data = try fixture(fixtureName)
            do {
                _ = try await client.send(ChatRequest(message: "Hello", history: []))
                XCTFail("Expected HTTP \(status) to fail")
            } catch let error as ChatAPIError {
                switch (status, error) {
                case (422, .invalidRequest), (429, .rateLimited), (502, .upstreamFailure), (503, .serviceUnavailable): break
                default: XCTFail("Unexpected mapping: \(error)")
                }
            }
        }
    }

    func testTimeoutAndCancellationAreNormalized() async {
        MockURLProtocol.error = URLError(.timedOut)
        await assertError(.timeout)
        MockURLProtocol.error = URLError(.cancelled)
        await assertError(.cancelled)
    }

    func testMalformedJSONIsNormalized() async throws {
        MockURLProtocol.data = try fixture("malformed")
        await assertError(.malformedResponse)
    }

    private func assertError(_ expected: ChatAPIError) async {
        do {
            _ = try await client.send(ChatRequest(message: "Hello", history: []))
            XCTFail("Expected failure")
        } catch let error as ChatAPIError {
            XCTAssertEqual(error, expected)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    private func fixture(_ name: String) throws -> Data {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: name, withExtension: "json"))
        return try Data(contentsOf: url)
    }
}

final class BackendSettingsTests: XCTestCase {
    func testNormalizationAddsSchemeAndTrimsSlashes() {
        XCTAssertEqual(BackendSettings.normalized(" 192.168.1.20:8000/ "), "http://192.168.1.20:8000")
        XCTAssertEqual(BackendSettings.normalized("abc.trycloudflare.com"), "https://abc.trycloudflare.com")
        XCTAssertEqual(BackendSettings.normalized("https://abc.trycloudflare.com/"), "https://abc.trycloudflare.com")
        XCTAssertEqual(BackendSettings.normalized("   "), BackendSettings.defaultURL)
    }

    func testLocalHostDetection() {
        XCTAssertTrue(BackendSettings.isLocalHost("127.0.0.1"))
        XCTAssertTrue(BackendSettings.isLocalHost("localhost"))
        XCTAssertTrue(BackendSettings.isLocalHost("my-mac.local"))
        XCTAssertTrue(BackendSettings.isLocalHost("10.0.0.7"))
        XCTAssertFalse(BackendSettings.isLocalHost("example.com"))
        XCTAssertFalse(BackendSettings.isLocalHost("999.1.1.1"))
    }

    func testPlainHTTPIsOnlyAllowedForLocalHosts() {
        XCTAssertThrowsError(try ChatAPIClient(baseURL: URL(string: "http://example.com")!))
        XCTAssertNoThrow(try ChatAPIClient(baseURL: URL(string: "http://192.168.1.5:8000")!))
        XCTAssertNoThrow(try ChatAPIClient(baseURL: URL(string: "https://abc.trycloudflare.com")!))
    }

    func testSavedURLWinsOverDefault() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "acharya-tests-\(UUID().uuidString)"))
        XCTAssertEqual(BackendSettings.currentURLString(defaults: defaults), BackendSettings.defaultURL)
        defaults.set("https://abc.trycloudflare.com", forKey: BackendSettings.storageKey)
        XCTAssertEqual(BackendSettings.currentURLString(defaults: defaults), "https://abc.trycloudflare.com")
    }
}

@MainActor
final class ChatViewModelTests: XCTestCase {
    func testSuccessCreatesCompletePairAndClearsInput() async throws {
        let provider = StubClient(results: [.success(Self.response)])
        let model = ChatViewModel(client: provider)
        model.message = "  hello  "
        model.sendMessage()
        XCTAssertEqual(model.pendingMessage, "hello")
        await waitUntil { !model.isWaitingForResponse }
        XCTAssertEqual(model.chatMessages.map(\.role), [.user, .assistant])
        XCTAssertEqual(model.message, "")
        XCTAssertNil(model.pendingMessage)
    }

    func testClearConversation() async throws {
        let model = ChatViewModel(client: StubClient(results: [.success(Self.response)]))
        model.message = "hello"
        model.sendMessage()
        await waitUntil { !model.isWaitingForResponse }
        model.clearConversation()
        XCTAssertTrue(model.chatMessages.isEmpty)
    }

    func testServerResponseWithModelAndLatencyDecodes() throws {
        let json = #"{"answer":"A","citations":[],"scores":[],"mode":"finetuned_rag","warning":null,"outcome":"answered","model":"ollama:acharyagpt","latency_ms":1234}"#
        let response = try JSONDecoder().decode(ChatResponse.self, from: Data(json.utf8))
        XCTAssertEqual(response.latencyMs, 1234)
        XCTAssertEqual(response.model, "ollama:acharyagpt")
    }

    func testFailureRetainsInputAndRetrySucceeds() async throws {
        let provider = StubClient(results: [.failure(ChatAPIError.serviceUnavailable("Not ready")), .success(Self.response)])
        let model = ChatViewModel(client: provider)
        model.message = "retained question"
        model.sendMessage()
        await waitUntil { !model.isWaitingForResponse }
        XCTAssertEqual(model.message, "retained question")
        XCTAssertNotNil(model.errorMessage)
        model.retry()
        await waitUntil { !model.isWaitingForResponse }
        XCTAssertEqual(model.chatMessages.count, 2)
        XCTAssertEqual(model.message, "")
    }

    func testViewModelSurfacesAllRecoverableFailuresAndRetainsInput() async {
        let failures: [ChatAPIError] = [
            .invalidRequest("Invalid"), .rateLimited("Limited"),
            .upstreamFailure("Upstream"), .serviceUnavailable("Unavailable"),
            .timeout, .malformedResponse,
        ]
        for failure in failures {
            let model = ChatViewModel(client: StubClient(results: [.failure(failure)]))
            model.message = "keep me"
            model.sendMessage()
            await waitUntil { !model.isWaitingForResponse }
            XCTAssertEqual(model.message, "keep me")
            XCTAssertNotNil(model.errorMessage)
            XCTAssertTrue(model.chatMessages.isEmpty)
        }
    }

    func testCancellationStopsRequestAndRetainsInput() async {
        let model = ChatViewModel(client: DelayedClient())
        model.message = "keep after cancel"
        model.sendMessage()
        model.cancel()
        await waitUntil { !model.isWaitingForResponse }
        XCTAssertEqual(model.message, "keep after cancel")
        XCTAssertEqual(model.errorMessage, ChatAPIError.cancelled.localizedDescription)
        XCTAssertTrue(model.chatMessages.isEmpty)
    }

    func testBlankAndDuplicateMessagesArePrevented() async throws {
        let provider = StubClient(results: [.success(Self.response)])
        let model = ChatViewModel(client: provider)
        model.message = "   "
        model.sendMessage()
        XCTAssertTrue(model.chatMessages.isEmpty)
        model.message = "same"
        model.sendMessage()
        await waitUntil { !model.isWaitingForResponse }
        model.message = "same"
        model.sendMessage()
        XCTAssertEqual(model.chatMessages.count, 2)
        XCTAssertNotNil(model.errorMessage)
    }

    private func waitUntil(_ condition: @escaping () -> Bool) async {
        for _ in 0..<100 where !condition() {
            try? await Task.sleep(for: .milliseconds(10))
        }
    }

    private static let response = ChatResponse(answer: "Answer", citations: [], scores: [], mode: "bm25_only", warning: nil, outcome: "answered")
}

private final class DelayedClient: ChatAPIClientProtocol {
    func send(_ request: ChatRequest) async throws -> ChatResponse {
        try await Task.sleep(for: .seconds(10))
        throw ChatAPIError.timeout
    }
}

private final class StubClient: ChatAPIClientProtocol {
    private var results: [Result<ChatResponse, Error>]

    init(results: [Result<ChatResponse, Error>]) { self.results = results }

    func send(_ request: ChatRequest) async throws -> ChatResponse {
        try results.removeFirst().get()
    }
}
