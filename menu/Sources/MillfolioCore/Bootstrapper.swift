import Foundation
import AppKit
import Darwin  // execv for the re-exec-after-self-upgrade hardening
import CryptoKit  // SHA256 verification of the downloaded source bundle

/// Drives the local engine lifecycle, as three explicit steps:
///
///   1. **Install server** — fetch the official Mojo compiler+runtime from
///      Modular's conda channel (so the *user* accepts Modular's license — we
///      never redistribute it), unpack our engine source zip (inference-server +
///      pkgs/jinja2.mojoc + flare + a prebuilt libflare_tls.so), build the server with
///      `mojo build`, then download the default model's weights with the engine's
///      own native-Mojo downloader (no huggingface_hub).
///   2. **Start server** — launch the built server (via a launchd LaunchAgent, so
///      the CLI and the menu app share one managed process).
///   3. **Start opencode** — point opencode at the running server (new Terminal).
///
/// Everything lives under ~/Library/Application Support/Millfolio, including the
/// model weights (HF_HOME=<support>/hf), so uninstall is a single directory.
///
/// This type is UI-agnostic on purpose: the menu-bar app observes it as an
/// `ObservableObject` (via `phase`/`serverRunning`), while the `mill` CLI
/// drives the same methods and streams progress through `onProgress`.
///
/// NOTE: the Mojo fetch is "rattler-by-URL" — we don't link the rattler crate, we
/// GET the pinned `.conda` packages (a .conda is a zip of zstd tarballs) and
/// extract them with the system `unzip`/`tar`. Keep `mojoVersion` in sync with
/// inference-server/pixi.lock.
@MainActor
public final class Bootstrapper: ObservableObject {
    public enum Phase: Equatable {
        case idle
        case running(String)
        case done
        case failed(String)

        public var message: String? {
            switch self {
            case .idle, .done: return nil
            case .running(let m): return m
            case .failed(let e): return "Failed: \(e)"
            }
        }
    }

    /// Progress of the long-running provisioning steps.
    @Published public var phase: Phase = .idle
    /// True while the engine server's LaunchAgent is loaded.
    @Published public var serverRunning = false

    /// Optional progress sink — every status message is forwarded here as well as
    /// to `phase`, so a non-UI driver (the CLI) can stream the same text.
    public var onProgress: ((String) -> Void)?

    public init() {
        // Off-main: a synchronous launchctl spawn here runs on the MAIN thread at
        // app launch (this class is @MainActor) — the first slice of the launch
        // beachball. The probe lands in `serverRunning` a beat later instead.
        refreshServerRunningInBackground()
    }

    public var isBusy: Bool { if case .running = phase { return true }; return false }

    // ── pinned manifest (keep in sync with inference-server/pixi.lock) ─────────────
    public static let mojoVersion = "1.0.0b3.dev2026062706"
    public static let condaChannel = "https://conda.modular.com/max-nightly"
    /// Default model served by the server. The 3B is int4-friendly and the
    /// quality target; its tokenizer.json is read directly by the engine.
    public static let model = "Qwen/Qwen2.5-3B-Instruct"
    public static let modelSlug = "Qwen--Qwen2.5-3B-Instruct"
    /// SECONDARY embedding model. The combined server resolves this from the HF
    /// cache to serve /v1/embeddings (else that endpoint 503s). millfolio's indexer
    /// + vault search hit it, so the installer fetches its weights too — via the
    /// same native-Mojo downloader, another HF id. Single-file safetensors (small).
    public static let embedModel = "Qwen/Qwen3-Embedding-0.6B"
    public static let embedModelSlug = "Qwen--Qwen3-Embedding-0.6B"
    // NOTE: model weights are NO LONGER downloaded at install time. The app server
    // provisions the embedding model + a default chat model in the background on
    // first start, and the UI catalog downloads any other model on demand (both via
    // `build/download`, whose path we hand the app server as MILLFOLIO_DOWNLOAD_BIN).
    // This keeps install fast and unable to fail on a multi-GB fetch.

    private var mojoCompilerURL: URL {
        URL(string: "\(Self.condaChannel)/osx-arm64/mojo-compiler-\(Self.mojoVersion)-release.conda")!
    }
    private var mojoPythonURL: URL {
        URL(string: "\(Self.condaChannel)/noarch/mojo-python-\(Self.mojoVersion)-release.conda")!
    }
    // ── enclave (privacy harness) ─────────────────────────────────────────────
    // enclave builds on the SAME unified Mojo toolchain as the server + vault —
    // every repo pins one nightly now, so it shares the single `mojoPrefix` toolchain
    // (no separate download). It's a one-shot CLI (not a daemon), so "start" opens a
    // ready-to-use Terminal rather than launching a server.
    public static let enclaveMojoVersion = "1.0.0b3.dev2026062706"
    private var enclaveMojoCompilerURL: URL {
        URL(string: "\(Self.condaChannel)/osx-arm64/mojo-compiler-\(Self.enclaveMojoVersion)-release.conda")!
    }
    private var enclaveMojoPythonURL: URL {
        URL(string: "\(Self.condaChannel)/noarch/mojo-python-\(Self.enclaveMojoVersion)-release.conda")!
    }
    /// Unified toolchain: enclave shares the single `mojoPrefix` install (the
    /// staleness check dedupes, so the toolchain is downloaded once for all components).
    private var enclaveMojoPrefix: URL { mojoPrefix }
    private var enclaveRoot: URL { bundleRoot.appendingPathComponent("enclave", isDirectory: true) }
    /// enclave checkout inside the unpacked bundle (sibling of flare/json/pkgs).
    private var enclaveDir: URL { enclaveRoot.appendingPathComponent("enclave", isDirectory: true) }
    private var enclaveBin: URL { enclaveDir.appendingPathComponent("build/enclave") }
    /// The built enclave binary is present.
    public var isEnclaveInstalled: Bool { FileManager.default.isExecutableFile(atPath: enclaveBin.path) }

    // ── millfolio (personal data vault) ───────────────────────────────────────────
    // millfolio is a one-shot vault CLI shipped PRECOMPILED (commercial IP
    // protection — no `.mojo` source on-device). Its bundle carries a prebuilt
    // build/millfolio binary, pkgs/*.mojoc (the vault tool surface + its libs,
    // precompiled against the SAME Mojo nightly as enclave), and the prebuilt
    // FFI shims. Install just places the binary + installMillfolioShims(); the
    // generated `from vault import *` programs compile against `-I pkgs`.
    private var millfolioMojoCompilerURL: URL {
        URL(string: "\(Self.condaChannel)/osx-arm64/mojo-compiler-\(Self.enclaveMojoVersion)-release.conda")!
    }
    private var millfolioMojoPythonURL: URL {
        URL(string: "\(Self.condaChannel)/noarch/mojo-python-\(Self.enclaveMojoVersion)-release.conda")!
    }
    /// Unified toolchain: the vault build shares the single `mojoPrefix` install too.
    private var millfolioMojoPrefix: URL { mojoPrefix }
    private var millfolioRoot: URL { bundleRoot.appendingPathComponent("millfolio", isDirectory: true) }
    /// millfolio checkout inside the unpacked bundle.
    private var millfolioDir: URL { millfolioRoot.appendingPathComponent("millfolio", isDirectory: true) }
    private var millfolioBin: URL { millfolioDir.appendingPathComponent("build/millfolio") }
    /// The built millfolio binary is present.
    public var isMillfolioInstalled: Bool { FileManager.default.isExecutableFile(atPath: millfolioBin.path) }

    // ── app server (the streaming WS backend, from millfolio/app) ──────────────
    // Built ON-DEVICE against the enclave engine tree, reusing enclave's Mojo
    // toolchain + flare shims — so no new toolchain. See app/server/CUTOVER.md.
    private var appRoot: URL { bundleRoot.appendingPathComponent("app", isDirectory: true) }
    private var appServerBin: URL { appRoot.appendingPathComponent("build/millfolio-server") }
    /// The built app server (UI + REST + the chat WS, all on one port) is present.
    public var isAppServerInstalled: Bool { FileManager.default.isExecutableFile(atPath: appServerBin.path) }

    // ── default config files (~/.config) ───────────────────────────────────────
    // Seeded with sensible defaults on install if absent, so a fresh setup has an
    // editable starting point. The engines read these (engine = the inference engine,
    // enclave = enclave); we NEVER overwrite an existing file.
    private var dotConfig: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".config", isDirectory: true)
    }
    private var engineConfigURL: URL { dotConfig.appendingPathComponent("millfolio/config.json") }
    private var enclaveConfigURL: URL { dotConfig.appendingPathComponent("enclave/config.json") }

    private static let engineConfigDefault = """
    {
      "port": 8000,
      "model": "Qwen/Qwen2.5-3B-Instruct",
      "q4": false,
      "kv_budget_mb": 8192
    }
    """
    private static let enclaveConfigDefault = """
    {
      "local_url": "http://127.0.0.1:8000/v1",
      "local_model": "Qwen2.5-0.5B-Instruct",
      "remote_base_url": "https://api.anthropic.com/v1",
      "remote_model": "claude-sonnet-5",
      "remote_token_budget": 200000,
      "mock": false,
      "use_local_summary": false,
      "data_dir": ""
    }
    """

    /// Create `path` with `json` if it doesn't exist (best-effort; never overwrites).
    private func ensureConfig(at path: URL, _ json: String) {
        let fm = FileManager.default
        guard !fm.fileExists(atPath: path.path) else { return }
        do {
            try fm.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try json.write(to: path, atomically: true, encoding: .utf8)
            appendLog("wrote default config: \(path.path)\n")
        } catch {
            appendLog("could not write config \(path.path): \(error)\n")  // non-fatal
        }
    }

    // ── install locations ─────────────────────────────────────────────────────
    private var support: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Millfolio", isDirectory: true)
    }
    private var mojoPrefix: URL { support.appendingPathComponent("mojo", isDirectory: true) }
    private var cacheDir: URL { support.appendingPathComponent("cache", isDirectory: true) }

    // ── source bundle (one millfolio.zip from the vault repo) ────────────────────
    // All on-device-built source ships in ONE archive whose subtrees mirror the
    // former four zips: runner/ enclave/ millfolio/ app/. Downloaded once and
    // built per-component, so the per-component build steps below are unchanged.
    // Pin the bundle to THIS CLI's release tag, not /releases/latest/, so the CLI and
    // its bundle move atomically AND a dev (pre-release) CLI fetches the dev bundle —
    // GitHub's /releases/latest skips pre-releases, so /latest/ would hand a dev CLI
    // the last PROD bundle. A source build with no Homebrew tag falls back to /latest/.
    /// The standalone `.app` (set by the menu-bar app) is NOT the versioned `mill`
    /// CLI, so it must NOT inherit whatever `mill` version happens to be brew-installed
    /// (that gave the app the old v0.4.36 bundle → "missing vault.mojoc"). When set, the
    /// app pins to the latest PROD release instead — its own cadence, always a current
    /// bundle format.
    public var forceLatestBundle = false

    /// For the standalone app (`forceLatestBundle`), the latest PROD release tag —
    /// resolved from GitHub's `/releases/latest` redirect (see `resolveProdLatestVersion`)
    /// and cached for the lifetime of this Bootstrapper. Empty until resolved, and left
    /// empty when resolution fails (offline) so the staleness check degrades to
    /// content-only (we never wipe a present bundle on a failed lookup). The CLI path
    /// ignores this — it pins to its Homebrew tag.
    private var resolvedProdLatest = ""

    /// The bundle version to pin to: the installed CLI's release (CLI), the resolved
    /// latest PROD tag (the standalone app, once `resolveProdLatestVersion` has run), or
    /// "" for an unmanaged build / the app before a successful prod-latest lookup.
    /// Empty → the staleness check in `ensureBundle` degrades to content-only.
    private func bundleVersion() -> String { forceLatestBundle ? resolvedProdLatest : brewCliVersion() }

    private var bundleURL: URL {
        // The standalone app always tracks the latest PROD release, so it fetches from
        // `/releases/latest/` (which serves the resolved prod-latest version's asset) —
        // NOT the versioned path. Keeping it at /latest/ means the URL is correct even
        // before `resolvedProdLatest` is known, and if a newer release is cut mid-run the
        // bundle sha256 check fails closed rather than installing a mismatched asset.
        if forceLatestBundle {
            return URL(string: "https://github.com/millfolio/vault/releases/latest/download/millfolio.zip")!
        }
        let v = bundleVersion()  // "v0.4.39" (CLI pin) or "" (unmanaged → /latest/)
        if !v.isEmpty {
            return URL(string: "https://github.com/millfolio/vault/releases/download/\(v)/millfolio.zip")!
        }
        return URL(string: "https://github.com/millfolio/vault/releases/latest/download/millfolio.zip")!
    }

    // ── prod-latest resolution (standalone app) ──────────────────────────────────
    /// Resolve the latest PROD release tag for `millfolio/vault` WITHOUT a Homebrew
    /// install or an API token: issue a `HEAD` to `…/releases/latest` and read the
    /// redirect's `Location`, which GitHub points at `…/releases/tag/vX.Y.Z` (the
    /// current latest NON-prerelease). Returns the tag (e.g. "v0.4.40"), or nil when the
    /// lookup fails (offline, DNS, an unexpected redirect target). Never throws —
    /// callers treat nil as "couldn't check" and keep the installed bundle. The redirect
    /// is deliberately preferred over the `releases/latest` API, which rate-limits
    /// unauthenticated requests.
    private func resolveProdLatestVersion() async -> String? {
        guard let url = URL(string:
            "https://github.com/millfolio/vault/releases/latest") else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "HEAD"
        req.timeoutInterval = 8
        req.cachePolicy = .reloadIgnoringLocalCacheData
        let catcher = RedirectCatcher()   // capture the Location, don't follow it
        guard let (_, resp) = try? await URLSession.shared.data(for: req, delegate: catcher),
              let http = resp as? HTTPURLResponse else { return nil }
        // With the redirect suppressed we see the 3xx and its Location. If a cache/proxy
        // instead handed us the followed 200, fall back to the response's final URL —
        // both end in `…/releases/tag/vX.Y.Z`.
        let target = http.value(forHTTPHeaderField: "Location") ?? http.url?.absoluteString
        guard let target, let tag = Self.releaseTag(fromReleasesURL: target) else { return nil }
        return tag
    }

    /// Pull `vX.Y.Z` out of a `…/releases/tag/vX.Y.Z` URL (GitHub's latest-redirect
    /// target). Validates the shape so a garbage redirect can't poison the pin.
    private static func releaseTag(fromReleasesURL s: String) -> String? {
        let path = URLComponents(string: s)?.path ?? s
        guard let last = path.split(separator: "/").last.map(String.init),
              isReleaseTag(last) else { return nil }
        return last
    }

    /// A plausible release tag: `v` + a digit, then digits / dots / dashes / letters
    /// (dashes+letters cover a `-rc.N` suffix defensively — `/releases/latest` never
    /// points at a pre-release, but validating loosely is safe and future-proof).
    private static func isReleaseTag(_ s: String) -> Bool {
        guard s.hasPrefix("v"), s.count > 1 else { return false }
        let body = s.dropFirst()
        guard body.first?.isNumber == true else { return false }
        return body.allSatisfy { $0.isNumber || $0 == "." || $0 == "-" || $0.isLetter }
    }
    private var bundleRoot: URL { support.appendingPathComponent("bundle", isDirectory: true) }
    private var engineRoot: URL { bundleRoot.appendingPathComponent("runner", isDirectory: true) }
    /// HF cache root for the model weights (HF_HOME). Self-contained under support/.
    private var hfHome: URL { support.appendingPathComponent("hf", isDirectory: true) }
    /// inference-server checkout inside the unpacked engine zip.
    private var backendDir: URL { engineRoot.appendingPathComponent("inference-server", isDirectory: true) }
    private var serverBin: URL { backendDir.appendingPathComponent("build/server") }
    /// The native-Mojo HF weights downloader, built at install time and RUN AT
    /// RUNTIME by the app server (not the installer). Its absolute path is handed to
    /// the app-server LaunchAgent as MILLFOLIO_DOWNLOAD_BIN.
    private var downloadBin: URL { backendDir.appendingPathComponent("build/download") }
    /// All subprocess output (mojo build, weights download, the running server)
    /// is appended here so errors that flash by in the menu can be read in full.
    public var logFileURL: URL { support.appendingPathComponent("Millfolio.log") }
    public var hasLog: Bool { FileManager.default.fileExists(atPath: logFileURL.path) }

    /// The built engine server binary is present.
    public var isServerInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: serverBin.path)
    }
    /// The default model's weights have been fully downloaded (refs/main is the
    /// downloader's last write, so its presence means the snapshot is complete).
    public var weightsPresent: Bool {
        FileManager.default.fileExists(
            atPath: hfHome.appendingPathComponent("hub/models--\(Self.modelSlug)/refs/main").path)
    }
    /// The embedding model's weights are fully downloaded (refs/main is the
    /// downloader's last write). When present, the combined server serves
    /// /v1/embeddings (so mill index/search work with no manual download).
    public var embedWeightsPresent: Bool {
        FileManager.default.fileExists(
            atPath: hfHome.appendingPathComponent("hub/models--\(Self.embedModelSlug)/refs/main").path)
    }
    /// Ready to launch: engine built and (chat) weights downloaded. The embedding
    /// weights are not required to start the chat server, so they don't gate this.
    public var canStartServer: Bool { isServerInstalled && weightsPresent && !serverRunning }

    /// Whether the full vault runtime is provisioned: all four on-device components
    /// (engine, vault tools, enclave harness, app server) are placed and
    /// executable. This is the "does the app need to run first-run onboarding?"
    /// signal — a read-only composite of the per-component `is…Installed` flags, so
    /// it never triggers work. Weights are NOT part of this: they're fetched at
    /// runtime by the app server (rc.21), and the web UI drives model/data choice.
    ///
    /// NOTE: this property (plus the `openBrowser` flag on `startAppServer`/
    /// `startVaultChat` below) is the ONLY addition to this file vs. the canonical
    /// `vault/cli` copy. Both are pure / behavior-preserving; upstream them to keep
    /// the two copies in sync.
    public var isProvisioned: Bool {
        isServerInstalled && isMillfolioInstalled && isEnclaveInstalled && isAppServerInstalled
    }

    // ── logging ──────────────────────────────────────────────────────────────
    /// Cap on the install log; past this we keep the (most-useful) tail. Bounds the
    /// file across repeated `mill install`/`mill update` runs.
    private static let maxLogBytes = 5 * 1024 * 1024   // 5 MB

    /// Ensure the log file (and its directory) exist; returns the path. When the log
    /// has grown past `maxLogBytes`, truncate it to its tail (recent lines are the ones
    /// worth keeping for diagnosis) behind a marker.
    @discardableResult
    private func ensureLog() -> URL {
        let fm = FileManager.default
        try? fm.createDirectory(at: support, withIntermediateDirectories: true)
        if !fm.fileExists(atPath: logFileURL.path) {
            fm.createFile(atPath: logFileURL.path, contents: nil)
        } else if let size = (try? fm.attributesOfItem(atPath: logFileURL.path)[.size]) as? Int,
                  size > Self.maxLogBytes,
                  let fh = try? FileHandle(forReadingFrom: logFileURL) {
            defer { try? fh.close() }
            fh.seek(toFileOffset: UInt64(max(0, size - Self.maxLogBytes / 2)))
            let tail = fh.readDataToEndOfFile()
            let header = Data("===== log truncated (was \(size / 1024) KB) — \(Self.stamp()) =====\n".utf8)
            try? (header + tail).write(to: logFileURL)
        }
        return logFileURL
    }

    /// Append text to the log (best-effort; never throws). Each line is prefixed with a
    /// `[HH:MM:SS.mmm]` timestamp — the same format as the Mojo `logging` library — so
    /// the log carries timing for every step and every subprocess line.
    private func appendLog(_ text: String) {
        ensureLog()
        guard let fh = try? FileHandle(forWritingTo: logFileURL) else { return }
        defer { try? fh.close() }
        fh.seekToEndOfFile()
        let ts = Self.tstamp()
        let stamped = text.components(separatedBy: "\n")
            .map { $0.isEmpty ? "" : "\(ts) \($0)" }
            .joined(separator: "\n")
        if let d = stamped.data(using: .utf8) { fh.write(d) }
    }

    private func logHeader(_ what: String) {
        appendLog("\n===== \(what) — \(Self.stamp()) =====\n")
    }

    /// `[HH:MM:SS.mmm]` per-line stamp — mirrors the Mojo `logging` library's format.
    private static func tstamp() -> String {
        let f = DateFormatter(); f.dateFormat = "[HH:mm:ss.SSS]"
        return f.string(from: Date())
    }

    private static func stamp() -> String {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f.string(from: Date())
    }

    /// Open the log in the user's default viewer (Console/TextEdit).
    public func openLog() {
        NSWorkspace.shared.open(ensureLog())
    }

    // ── millfolio diagnostic log (~/Library/Logs/Millfolio/<date>.log) ──────────────
    // Separate from Millfolio.log: a per-day, user-facing diagnostic log for the
    // `millfolio` CLI itself (the ask/index runs + update), in the conventional
    // macOS ~/Library/Logs location so it's easy to find and attach to a report.
    public var millfolioLogDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Millfolio", isDirectory: true)
    }
    /// Today's log file, e.g. ~/Library/Logs/Millfolio/2026-06-17.log.
    public var millfolioLogURL: URL {
        millfolioLogDir.appendingPathComponent("\(Self.day()).log")
    }

    @discardableResult
    private func ensureMillfolioLog() -> URL {
        let fm = FileManager.default
        try? fm.createDirectory(at: millfolioLogDir, withIntermediateDirectories: true)
        if !fm.fileExists(atPath: millfolioLogURL.path) {
            fm.createFile(atPath: millfolioLogURL.path, contents: nil)
        }
        return millfolioLogURL
    }

    /// Append a line to today's millfolio log (best-effort; never throws).
    public func vlog(_ text: String) {
        ensureMillfolioLog()
        guard let fh = try? FileHandle(forWritingTo: millfolioLogURL) else { return }
        defer { try? fh.close() }
        fh.seekToEndOfFile()
        if let d = (text + "\n").data(using: .utf8) { fh.write(d) }
    }

    private static func day() -> String {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd"
        return f.string(from: Date())
    }

    // ── toolchain freshness ─────────────────────────────────────────────────────
    /// True if the Mojo toolchain at `prefix` is absent OR not the pinned `version`
    /// (recorded in a .mojo-version marker). Presence alone isn't enough: a stale
    /// nightly kept across a version bump can segfault on a newer macOS — so a
    /// version mismatch forces a re-download.
    private func mojoToolchainStale(_ prefix: URL, _ version: String) -> Bool {
        guard FileManager.default.isExecutableFile(
            atPath: prefix.appendingPathComponent("bin/mojo").path) else { return true }
        let have = (try? String(contentsOf: prefix.appendingPathComponent(".mojo-version"),
                                 encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return have != version
    }

    private func recordMojoVersion(_ prefix: URL, _ version: String) {
        try? version.write(to: prefix.appendingPathComponent(".mojo-version"),
                           atomically: true, encoding: .utf8)
    }

    // ── granular per-step install guards (existence AND freshness) ───────────────
    // The coarse "is the binary present?" skip is wrong in two ways: it leaves a
    // step skipped when a SHARED-toolchain re-provision wiped its lib/ (the missing
    // liblancedbmojo.dylib that crashed `mill index` after v0.4.35), and it skips a
    // REBUILD when a new release shipped new source under the same toolchain. So a
    // step is "current" iff EVERY critical artifact it produces still exists AND a
    // `.{step}` stamp equals the installed CLI version. Either a missing file or a
    // version bump re-triggers the step. Cheap: a handful of `fileExists` + one read.

    /// Cached Homebrew CLI version — the key below reads it repeatedly during an
    /// install and `brewCliVersion()` shells out, so resolve it once, lazily.
    private lazy var cachedBrewCliVersion: String = brewCliVersion()

    /// The freshness key: the installed `mill` CLI version (so a release bump re-runs
    /// every step), falling back to the pinned mojo nightly when brew can't be queried.
    /// For the standalone app (which has no Homebrew tag) it's the resolved latest PROD
    /// tag instead — so a bundle refresh to a newer PROD release actually re-fires every
    /// per-step guard (a constant mojo-version key would skip the rebuild and leave the
    /// engine compiled against the OLD source). The CLI path is byte-identical to before.
    private var installVersionKey: String {
        if forceLatestBundle, !resolvedProdLatest.isEmpty { return resolvedProdLatest }
        return cachedBrewCliVersion.isEmpty ? Self.mojoVersion : cachedBrewCliVersion
    }

    /// True when every `files` artifact exists AND the `.{stamp}` step marker
    /// matches the current install version. Pair with `recordStep` after install.
    private func stepCurrent(_ stamp: String, _ files: [URL]) -> Bool {
        let fm = FileManager.default
        for f in files where !fm.fileExists(atPath: f.path) { return false }
        let have = (try? String(contentsOf: support.appendingPathComponent(stamp),
                                 encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return have == installVersionKey
    }
    private func recordStep(_ stamp: String) {
        try? installVersionKey.write(to: support.appendingPathComponent(stamp),
                                     atomically: true, encoding: .utf8)
    }

    // ── step 1: install server (+ weights) ──────────────────────────────────────
    /// Menu-app entry point: fire-and-forget, drives `phase`. The CLI calls the
    /// throwing `installServer()` directly.
    public func downloadServer() {
        guard !isBusy else { return }
        phase = .running("Starting…")
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.installServer(); await self.set(done: true) }
            catch { await self.set(failed: humanError(error)) }
        }
    }

    // ── one-time migration: millrace → millfolio (pre-rename installs) ───────────
    /// Older versions installed under `~/Library/Application Support/Millrace`, with
    /// config/cache under `~/.config/millrace` + `~/.cache/millrace` and a
    /// `me.millrace.server` LaunchAgent. Move each to its `millfolio` location once,
    /// so upgraders keep their multi-GB model weights instead of re-downloading.
    ///
    /// Idempotent and best-effort: a path is moved only when the legacy location
    /// exists and the new one does not, so re-running (or a fresh install) is a
    /// no-op. After moving the tree, the stale engine *build* is dropped (weights
    /// under `hf/` are kept) so `installServer` rebuilds the binary against the new
    /// source + `~/.config/millfolio` config path.
    public func migrateLegacyLayout() {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser
        let appSup = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]

        // Boot out + remove the old LaunchAgent first; the new install writes the
        // me.millfolio.server agent, and leaving the old one loaded runs a stale
        // binary against a config path that no longer exists.
        let oldAgent = home.appendingPathComponent("Library/LaunchAgents/me.millrace.server.plist")
        if fm.fileExists(atPath: oldAgent.path) {
            _ = try? run("/bin/launchctl", ["bootout", "gui/\(getuid())/me.millrace.server"])
            try? fm.removeItem(at: oldAgent)
            appendLog("migrated: removed legacy LaunchAgent me.millrace.server\n")
        }

        let legacyTree = appSup.appendingPathComponent("Millrace", isDirectory: true)
        let migratedTree = fm.fileExists(atPath: legacyTree.path)
            && !fm.fileExists(atPath: support.path)

        let moves: [(URL, URL)] = [
            (legacyTree, support),
            (home.appendingPathComponent(".config/millrace", isDirectory: true),
             home.appendingPathComponent(".config/millfolio", isDirectory: true)),
            (home.appendingPathComponent(".cache/millrace", isDirectory: true),
             home.appendingPathComponent(".cache/millfolio", isDirectory: true)),
        ]
        for (old, new) in moves where fm.fileExists(atPath: old.path) && !fm.fileExists(atPath: new.path) {
            do {
                try fm.createDirectory(at: new.deletingLastPathComponent(), withIntermediateDirectories: true)
                try fm.moveItem(at: old, to: new)
                appendLog("migrated: \(old.path) → \(new.path)\n")
            } catch {
                appendLog("migration skipped for \(old.lastPathComponent): \(humanError(error))\n")
            }
        }

        guard migratedTree else { return }
        // Keep the model weights (hf/) + cache; drop the unpacked source bundle so it
        // re-downloads + rebuilds fresh against the new layout + ~/.config/millfolio.
        try? fm.removeItem(at: bundleRoot)
        // The per-day diagnostic log inside the moved tree kept its old name.
        let oldLog = support.appendingPathComponent("Millrace.log")
        if fm.fileExists(atPath: oldLog.path) && !fm.fileExists(atPath: logFileURL.path) {
            try? fm.moveItem(at: oldLog, to: logFileURL)
        }
    }

    /// Provision the Mojo toolchain, engine source, build, and weights. Throws on
    /// the first failure (the CLI surfaces it; the menu wrapper maps it to `phase`).
    public func installServer() async throws {
        migrateLegacyLayout()   // upgrade an older millrace-layout install in place
        // Idempotent fast-path: the engine server + the weights downloader are built
        // and the toolchain is current → nothing to do. Weights are NOT part of the
        // install anymore (the app server provisions them at runtime), so they don't
        // gate this.
        if stepCurrent(".engine-step", [serverBin, downloadBin])
            && !mojoToolchainStale(mojoPrefix, Self.mojoVersion) {
            set("engine already installed — skipping")
            return
        }
        let fm = FileManager.default
        for d in [support, mojoPrefix, engineRoot, cacheDir, hfHome] {
            try fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
        logHeader("Install server")

        if mojoToolchainStale(mojoPrefix, Self.mojoVersion) {
            set("Downloading Mojo compiler for engine (~70 MB)…")
            try? fm.removeItem(at: mojoPrefix)   // clear any stale nightly
            try fm.createDirectory(at: mojoPrefix, withIntermediateDirectories: true)
            let compiler = try await download(mojoCompilerURL, name: "mojo-compiler.conda")
            set("Extracting Mojo…")
            try extractConda(compiler, into: mojoPrefix)
            let py = try await download(mojoPythonURL, name: "mojo-python.conda")
            try extractConda(py, into: mojoPrefix)
            recordMojoVersion(mojoPrefix, Self.mojoVersion)
        }
        try relocateMojoPrefix(mojoPrefix)   // rewrite modular.cfg's baked placeholder prefix

        try await ensureBundle()

        // The engine ships as SOURCE and is compiled ON-DEVICE (unlike enclave +
        // the app server, which now ship prebuilt): its AOT GPU/Metal kernels can't
        // be built on the GPU-less GitHub CI runner ("Unknown GPU architecture
        // detected"), so the compile must happen on the user's Mac. That's the same
        // reason GPU gates only ever run in local preflight, never in CI.
        let python = try findPython()

        set("Building engine (first run, ~1 min)…")
        try buildBinary(python: python, source: "src/server.mojo",
                        args: ["-I", "../pkgs", "-I", "../flare"], out: "build/server")
        signServerIdentity()

        // Build the native-Mojo weights downloader — NOT to download here, but so the
        // app server can run it at runtime (MILLFOLIO_DOWNLOAD_BIN) to provision the
        // embedding + default chat model in the background and to fulfil the UI
        // catalog's on-demand downloads. Install ships binaries + toolchain only, so
        // it stays fast and can't fail on a multi-GB fetch.
        if !FileManager.default.isExecutableFile(atPath: downloadBin.path) {
            set("Building weights downloader…")
            try buildBinary(python: python, source: "src/download.mojo",
                            args: ["-I", "../flare"], out: "build/download")
        }

        ensureConfig(at: engineConfigURL, Self.engineConfigDefault)
        recordStep(".engine-step")
    }

    // ── step 2: start / stop server (launchd LaunchAgent) ────────────────────────
    // The server runs as a per-user LaunchAgent (me.millfolio.server) instead of a
    // child Process, so the `mill` CLI and the menu app's "Start
    // server" drive the SAME managed process — either surface can start/stop/see it.
    public static let serverLabel = "me.millfolio.server"
    private var launchAgentURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents/\(Self.serverLabel).plist")
    }
    // The app server (UI + REST + chat WS on :10000) now runs under launchd too —
    // the SAME mechanism as the inference server — instead of a nohup that orphaned
    // and raced the port. Both are launchd agents; the only difference is which env
    // they carry. (They stay TWO processes so the app can restart without reloading
    // the ~7GB model.)
    public static let appServerLabel = "me.millfolio.appserver"
    private var appServerLaunchAgentURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents/\(Self.appServerLabel).plist")
    }
    private var guiDomain: String { "gui/\(getuid())" }

    /// Start the server LaunchAgent. Idempotent: re-bootstraps a fresh plist.
    /// Async — the launchctl spawns run off-main (see `spawnProcess`).
    public func startServer() async throws {
        // Weights are provisioned at runtime by the app server (not at install), so
        // this no longer gates on them: we bootstrap the engine agent even with no
        // weights yet, so the app server's provisioner can `launchctl kickstart` it
        // the moment the default model finishes downloading. The engine reads its
        // model from config.json (single source of truth).
        guard isServerInstalled else {
            throw BootstrapError.step("start server", "engine not installed — run install first")
        }
        try writeLaunchAgent()
        logHeader("Start server: \(Self.model)")
        // Replace any prior instance, then load (RunAtLoad starts it).
        _ = await runStatusAsync("/bin/launchctl", ["bootout", "\(guiDomain)/\(Self.serverLabel)"])
        try await runAsync("/bin/launchctl", ["bootstrap", guiDomain, launchAgentURL.path])
        serverRunning = true
    }

    /// Stop the server LaunchAgent (no-op if not loaded).
    public func stopServer() async {
        let rc = await runStatusAsync("/bin/launchctl", ["bootout", "\(guiDomain)/\(Self.serverLabel)"])
        if rc != 0 { appendLog("[launchctl bootout exited \(rc) — not loaded?]\n") }
        serverRunning = false
    }

    /// Non-throwing menu-button wrappers: surface any failure via `phase`.
    public func tryStartServer() {
        Task { do { try await startServer() } catch { phase = .failed(humanError(error)) } }
    }
    public func tryStopServer() {
        Task { await stopServer() }
    }

    /// Reconcile `serverRunning` with launchd's actual state (e.g. at app launch).
    public func refreshServerRunning() {
        let loaded = (try? runStatus("/bin/launchctl", ["print", "\(guiDomain)/\(Self.serverLabel)"])) == 0
        serverRunning = loaded
    }

    /// The non-blocking variant the APP uses (init + the menu's Refresh): probe
    /// launchd on a background queue, publish the result back on the main actor.
    /// The synchronous `refreshServerRunning()` stays for the CLI-style paths that
    /// need the answer before their next line.
    public func refreshServerRunningInBackground() {
        Task { [weak self] in
            guard let self else { return }
            let code = await Self.spawnProcess(
                "/bin/launchctl", ["print", "\(self.guiDomain)/\(Self.serverLabel)"]).code
            self.serverRunning = (code == 0)
        }
    }

    /// Write the LaunchAgent plist that runs the built server against the weights.
    private func writeLaunchAgent() throws {
        try FileManager.default.createDirectory(
            at: launchAgentURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        // Minimal explicit env — launchd does NOT inherit the app's environment.
        // Keep CONDA_PREFIX unset so flare loads build/libflare_tls.so next to the
        // binary; HOME is provided by launchd (kv-cache lives under ~/.cache).
        var env: [String: String] = [
            "HF_HOME": hfHome.path,
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        ]
        if FileManager.default.fileExists(atPath: "/etc/ssl/cert.pem") {
            env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
        }
        let plist: [String: Any] = [
            "Label": Self.serverLabel,
            // No model arg: the engine reads its model from config.json (engineConfigURL,
            // default Self.model), making that file the single source of truth. The UI's
            // on-device model selector switches models by rewriting config.model and
            // kicking this agent — which only works if the model isn't pinned here as argv[1].
            "ProgramArguments": [serverBin.path],
            "WorkingDirectory": backendDir.path,   // hardcoded relative data paths resolve here
            "EnvironmentVariables": env,
            "StandardOutPath": logFileURL.path,
            "StandardErrorPath": logFileURL.path,
            "RunAtLoad": true,
            "ProcessType": "Interactive",
        ]
        let data = try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
        try data.write(to: launchAgentURL)
    }

    // ── step 3: start opencode ──────────────────────────────────────────────────
    /// Generate an opencode config from the running server's /v1/models, then open
    /// opencode in a new Terminal window pointed at it. opencode is an interactive
    /// TUI, so it must run in a real terminal, not detached.
    public func startOpencode() {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.launchOpencode() }
            catch { await self.set(failed: "opencode: \(humanError(error))") }
        }
    }

    public func launchOpencode() async throws {
        let base = "http://127.0.0.1:8000/v1"
        let opencode = try findOpencode()
        let configPath = try await writeOpencodeConfig(baseURL: base)

        // A small launcher script avoids AppleScript quoting pitfalls.
        let script = support.appendingPathComponent("run-opencode.sh")
        let body = """
        #!/bin/bash
        export OPENCODE_CONFIG="\(configPath)"
        export OPENAI_BASE_URL="\(base)"
        export OPENAI_API_KEY="millfolio"
        # opencode's own dir + common bins on PATH (Terminal already sources the
        # user's profile, but be explicit in case it shells out to helpers).
        export PATH="\(URL(fileURLWithPath: opencode).deletingLastPathComponent().path):$PATH"
        exec "\(opencode)"
        """
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        // `do script` runs its text as a shell command line, so the script path
        // (which lives under "Application Support" — note the space) must be shell-
        // quoted, or zsh splits it at the space. Single-quote it (the path has no
        // single quotes).
        let cmd = "'\(script.path)'"
        try run("/usr/bin/osascript",
                ["-e", "tell application \"Terminal\" to activate",
                 "-e", "tell application \"Terminal\" to do script \"\(cmd)\""])
    }

    /// Build the opencode provider config the way inference-server/opencode_config.py
    /// does, but in-process (no Python): query /v1/models and declare each served id.
    private func writeOpencodeConfig(baseURL: String) async throws -> String {
        guard let url = URL(string: baseURL + "/models") else {
            throw BootstrapError.step("opencode", "bad base URL")
        }
        var req = URLRequest(url: url); req.timeoutInterval = 3
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let arr = json["data"] as? [[String: Any]] else {
            throw BootstrapError.step("opencode", "server not reachable at \(baseURL)/models — start the server first")
        }
        let ids = arr.compactMap { $0["id"] as? String }
        guard let first = ids.first else { throw BootstrapError.step("opencode", "no models served") }
        var models: [String: Any] = [:]
        for id in ids { models[id] = ["name": id.components(separatedBy: "/").last ?? id] }
        let config: [String: Any] = [
            "$schema": "https://opencode.ai/config.json",
            "model": "millfolio/" + first,
            "provider": ["millfolio": [
                "npm": "@ai-sdk/openai-compatible",
                "name": "millfolio (local)",
                "options": ["baseURL": baseURL, "apiKey": "millfolio"],
                "models": models,
            ]],
        ]
        let out = cacheDir.appendingPathComponent("opencode.json")
        let blob = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted])
        try blob.write(to: out)
        return out.path
    }

    // ── steps ────────────────────────────────────────────────────────────────
    private func sha256Hex(of file: URL) throws -> String {
        // mappedIfSafe keeps a multi-hundred-MB bundle off the heap.
        let data = try Data(contentsOf: file, options: .mappedIfSafe)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Download `url` to `cacheDir/name`. When `expectedSHA256` is non-nil the
    /// bytes are verified BEFORE they are moved into place / used — a mismatch
    /// (tampered or corrupted download) throws and nothing is installed.
    private func download(_ url: URL, name: String, expectedSHA256: String? = nil) async throws -> URL {
        let dest = cacheDir.appendingPathComponent(name)
        let (tmp, resp) = try await URLSession.shared.download(from: url)
        guard (resp as? HTTPURLResponse).map({ (200..<300).contains($0.statusCode) }) ?? false else {
            throw BootstrapError.step("download \(name)", "HTTP error fetching \(url.absoluteString)")
        }
        if let want = expectedSHA256?.lowercased(), !want.isEmpty {
            let got = (try sha256Hex(of: tmp)).lowercased()
            guard got == want else {
                try? FileManager.default.removeItem(at: tmp)
                throw BootstrapError.step(
                    "verify \(name)",
                    "sha256 mismatch (expected \(want), got \(got)). Refusing to install a "
                        + "tampered or corrupted download.")
            }
        }
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
        return dest
    }

    /// The expected sha256 of `millfolio.zip` for `version`, fetched from the
    /// Homebrew TAP (`millfolio/homebrew-tap`) over HTTPS — deliberately NOT from
    /// the release that serves the zip. The tap is the same trust root `brew`
    /// already uses to verify the CLI tarball's own sha256, and it's a *different*
    /// repository from the GitHub release assets, so an attacker who can swap a
    /// release asset cannot also forge this hash. Returns nil when no checksum is
    /// published for `version` (releases cut before this landed, or a source /
    /// unmanaged install with no brew version) — `ensureBundle` decides whether
    /// that is allowed to proceed.
    private func expectedBundleSHA256(for version: String) async -> String? {
        guard !version.isEmpty, let url = URL(string:
            "https://raw.githubusercontent.com/millfolio/homebrew-tap/HEAD/checksums/millfolio-\(version).sha256")
        else { return nil }
        guard let (data, resp) = try? await URLSession.shared.data(from: url),
              (resp as? HTTPURLResponse).map({ $0.statusCode == 200 }) ?? false,
              let text = String(data: data, encoding: .utf8) else { return nil }
        // Accept a bare hex digest or `shasum`/`sha256sum` output (`<hex>  file`).
        let tok = text.split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" })
            .first.map(String.init)?.lowercased() ?? ""
        return tok.count == 64 && tok.allSatisfy { $0.isHexDigit } ? tok : nil
    }

    /// Download + unpack the one source bundle (millfolio.zip) once. Each component
    /// (server/enclave/millfolio/app) calls this before building, so it runs once
    /// per install and is a no-op thereafter. Gates on the **actual unpacked engine
    /// source**, not a stamp: a stamp can outlive its content (e.g. a stale `mill
    /// update` that removed runner/), which would skip the re-unpack and then fail
    /// later on a missing inference-server. `unpackZip` sanity-checks the subtree.
    private func ensureBundle() async throws {
        let fm = FileManager.default
        let stampURL = bundleRoot.appendingPathComponent(".bundle-version")
        let contentPresent = fm.fileExists(
            atPath: backendDir.appendingPathComponent("src/server.mojo").path)
        // The release this CLI belongs to (from Homebrew). The bundle is `releases/latest`,
        // so after `brew upgrade mill` this changes and the on-disk bundle is stale even
        // though its content is still "present" — re-fetch instead of forcing the user to
        // delete it by hand. Empty (non-brew install) → fall back to content-only so we
        // don't loop re-downloading with no version signal.
        // Standalone app: learn the latest PROD tag so `want` is a real version to
        // compare against the on-disk stamp (the CLI pins to its brew tag instead). A
        // failed lookup (offline) leaves resolvedProdLatest empty → want stays "" →
        // content-only staleness below (we never wipe a present bundle on a failed
        // lookup). Resolve once per Bootstrapper.
        if forceLatestBundle && resolvedProdLatest.isEmpty {
            if let tag = await resolveProdLatestVersion() { resolvedProdLatest = tag }
        }
        let want = bundleVersion()  // CLI tag, resolved prod-latest (app), or "" (offline/unmanaged) → /latest/
        let have = (try? String(contentsOf: stampURL, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if contentPresent && (want.isEmpty || have == want) { return }
        set(contentPresent && have != want
            ? "Refreshing millfolio sources for \(want)…" : "Downloading millfolio sources…")
        // Verify the bundle's sha256 against the tap-published checksum before we
        // unpack + COMPILE its contents (a tampered zip = arbitrary code execution
        // at build time). Fail closed when the release publishes a checksum; for
        // versions with none yet (pre-checksum releases) or a source install, warn
        // and proceed unless MILLFOLIO_REQUIRE_BUNDLE_CHECKSUM=1 demands strictness.
        let expected = await expectedBundleSHA256(for: want)
        if expected == nil && !want.isEmpty {
            if ProcessInfo.processInfo.environment["MILLFOLIO_REQUIRE_BUNDLE_CHECKSUM"] == "1" {
                throw BootstrapError.step(
                    "verify millfolio.zip",
                    "no published checksum for \(want) in the tap, and "
                        + "MILLFOLIO_REQUIRE_BUNDLE_CHECKSUM=1")
            }
            set("Warning: no published checksum for \(want) — installing unverified sources.")
        }
        let zip = try await download(bundleURL, name: "millfolio.zip", expectedSHA256: expected)
        if expected != nil { set("Sources verified (sha256 ✓). Unpacking…") }
        else { set("Unpacking sources…") }
        // Wipe the old tree first so a version change can't leave stale files behind
        // (unzip -o only overwrites; it never deletes files dropped from the new bundle).
        try? fm.removeItem(at: bundleRoot)
        try fm.createDirectory(at: bundleRoot, withIntermediateDirectories: true)
        try unpackZip(zip, into: bundleRoot)
        try? want.write(to: stampURL, atomically: true, encoding: .utf8)
    }

    /// A `.conda` is a zip containing `pkg-*.tar.zst` (the files) + `info-*.tar.zst`.
    /// We unzip it (native), zstd-decompress each payload IN-PROCESS via the
    /// vendored decoder, then untar the resulting plain `.tar`. The two-step
    /// avoids `tar`'s zstd filter, which on macOS shells out to a `zstd` program
    /// that isn't installed (libarchive here is built without built-in zstd).
    private func extractConda(_ conda: URL, into prefix: URL) throws {
        let scratch = cacheDir.appendingPathComponent("conda-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        try run("/usr/bin/unzip", ["-o", "-q", conda.path, "-d", scratch.path])
        let entries = try FileManager.default.contentsOfDirectory(atPath: scratch.path)
        let pkgs = entries.filter { $0.hasPrefix("pkg-") && $0.hasSuffix(".tar.zst") }
        guard !pkgs.isEmpty else { throw BootstrapError.step("extract", "no pkg tar in \(conda.lastPathComponent)") }
        for pkg in pkgs {
            let zst = scratch.appendingPathComponent(pkg)
            let tar = scratch.appendingPathComponent(String(pkg.dropLast(4)))   // strip ".zst"
            try Zstd.decompressFile(zst, to: tar)
            // Plain (uncompressed) tar — core libarchive, no optional filter.
            try run("/usr/bin/tar", ["-xf", tar.path, "-C", prefix.path])
        }
    }

    private func unpackZip(_ zip: URL, into dir: URL) throws {
        try run("/usr/bin/unzip", ["-o", "-q", zip.path, "-d", dir.path])
        guard FileManager.default.fileExists(atPath: backendDir.appendingPathComponent("src/server.mojo").path) else {
            throw BootstrapError.step("unpack", "engine zip missing inference-server/src/server.mojo")
        }
    }

    /// Find an existing Python >= 3.10 on the system (we do NOT download one).
    private func findPython() throws -> URL {
        let candidates = ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
            + (ProcessInfo.processInfo.environment["PATH"]?.split(separator: ":").map { String($0) + "/python3" } ?? [])
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            if let v = try? run(path, ["-c", "import sys;print(sys.version_info[0],sys.version_info[1])"]) {
                let parts = v.split(separator: " ").compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                if parts.count == 2, parts[0] == 3, parts[1] >= 10 { return URL(fileURLWithPath: path) }
            }
        }
        throw BootstrapError.step("python", "no Python >= 3.10 found on PATH (Mojo needs one; install one or add it to PATH)")
    }

    private func findOpencode() throws -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        // A GUI app's PATH is minimal and excludes per-user install dirs, so check
        // the common ones explicitly (opencode installs to ~/.opencode/bin).
        let candidates = [
            "\(home)/.opencode/bin/opencode",
            "\(home)/.local/bin/opencode",
            "/opt/homebrew/bin/opencode",
            "/usr/local/bin/opencode",
        ] + (ProcessInfo.processInfo.environment["PATH"]?.split(separator: ":").map { String($0) + "/opencode" } ?? [])
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) { return path }
        throw BootstrapError.step("opencode", "opencode not found — install it (https://opencode.ai) or add it to PATH")
    }

    /// Env for invoking `mojo build`. What conda's activation script exports — the
    /// compiler reads $MODULAR_HOME/modular.cfg for its stdlib import path + libs.
    private func mojoEnv(python: URL) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let extraPath = "\(python.deletingLastPathComponent().path):\(mojoPrefix.appendingPathComponent("bin").path)"
        env["PATH"] = extraPath + ":" + (env["PATH"] ?? "/usr/bin:/bin")
        env["CONDA_PREFIX"] = mojoPrefix.path
        env["MODULAR_HOME"] = mojoPrefix.appendingPathComponent("share/max").path
        return env
    }

    /// Env for *running* the compiled Mojo binaries (download) — the opposite of
    /// the build env: keep CONDA_PREFIX unset so flare loads `build/libflare_tls.so`
    /// next to the binary (cwd) rather than `$CONDA_PREFIX/lib`, and point OpenSSL
    /// at the system CA bundle (the bundled libssl's compiled-in cert path is the
    /// CI prefix, which is absent here).
    private func runtimeEnv() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "CONDA_PREFIX")
        env.removeValue(forKey: "MODULAR_HOME")
        if FileManager.default.fileExists(atPath: "/etc/ssl/cert.pem") {
            env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
        }
        return env
    }

    /// conda packages bake a placeholder install path into `share/max/modular.cfg`
    /// (the value of `package_root`), normally rewritten by conda's prefix-
    /// replacement step — which we skip by extracting the `.conda` by hand. Rewrite
    /// it to our real prefix so the compiler can locate the stdlib (`import_path`)
    /// and link the runtime libs (rpath). Idempotent; safe to run every time.
    private func relocateMojoPrefix(_ prefix: URL) throws {
        let cfg = prefix.appendingPathComponent("share/max/modular.cfg")
        guard var text = try? String(contentsOf: cfg, encoding: .utf8) else {
            throw BootstrapError.step("relocate", "modular.cfg missing after extract")
        }
        guard let line = text.split(separator: "\n").first(where: { $0.hasPrefix("package_root") }),
              let eq = line.firstIndex(of: "=") else {
            throw BootstrapError.step("relocate", "no package_root in modular.cfg")
        }
        let placeholder = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
        guard !placeholder.isEmpty, placeholder != prefix.path else { return }  // already done
        text = text.replacingOccurrences(of: placeholder, with: prefix.path)
        try text.write(to: cfg, atomically: true, encoding: .utf8)
        appendLog("relocated mojo prefix: \(placeholder) -> \(prefix.path)\n")
    }

    private func buildBinary(python: URL, source: String, args: [String], out: String) throws {
        let mojo = mojoPrefix.appendingPathComponent("bin/mojo").path
        // flare's libflare_tls.so ships at inference-server/build/ relative to cwd.
        try run(mojo, ["build", source] + args + ["-o", out], cwd: backendDir, env: mojoEnv(python: python))
    }

    /// `mojo build` ad-hoc "linker-signs" the server with the identifier "server".
    /// macOS's "<name> can run in the background" notification + Login Items entry
    /// for the LaunchAgent take that signing identifier as the name, so re-sign it
    /// (still ad-hoc) as "millfolio". Best-effort — purely cosmetic, so a failure
    /// never blocks the install.
    private func signServerIdentity() {
        do {
            try run("/usr/bin/codesign",
                    ["--force", "--sign", "-", "--identifier", "millfolio", serverBin.path])
        } catch {
            appendLog("could not re-sign server identity (cosmetic): \(humanError(error))\n")
        }
    }

    // Note: enclave + the app server are no longer `mojo build`d on-device —
    // they ship prebuilt (rpath-relocated + ad-hoc signed) by their CI packagers,
    // so install just verifies + chmods them. Only the engine still builds
    // on-device (its GPU/Metal kernels can't compile on the GPU-less CI runner),
    // which is why buildBinary/mojoEnv/signServerIdentity above are retained.

    // ── enclave: install ──────────────────────────────────────────────────────
    /// Menu-app entry point: fire-and-forget, drives `phase`.
    public func installEnclave() {
        guard !isBusy else { return }
        phase = .running("Starting…")
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.installEnclaveEngine(); await self.set(done: true) }
            catch { await self.set(failed: humanError(error)) }
        }
    }

    /// Download enclave's Mojo toolchain + source bundle and build it. Separate
    /// from the server: enclave is on a different nightly and ships its own
    /// vendored flare/json + compiled pkgs + prebuilt FFI shims.
    public func installEnclaveEngine() async throws {
        // Idempotent: skip the whole download+build if the binary is already there.
        if stepCurrent(".enclave-step", [enclaveBin])
            && !mojoToolchainStale(enclaveMojoPrefix, Self.enclaveMojoVersion) {
            set("enclave already installed — skipping")
            return
        }
        let fm = FileManager.default
        for d in [support, enclaveMojoPrefix, enclaveRoot, cacheDir] {
            try fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
        logHeader("Install enclave")

        // 1. Mojo toolchain (enclave's nightly — distinct from the engine's).
        if mojoToolchainStale(enclaveMojoPrefix, Self.enclaveMojoVersion) {
            set("Downloading Mojo compiler for enclave (~70 MB)…")
            try? fm.removeItem(at: enclaveMojoPrefix)   // clear any stale nightly
            try fm.createDirectory(at: enclaveMojoPrefix, withIntermediateDirectories: true)
            let compiler = try await download(enclaveMojoCompilerURL, name: "enclave-mojo-compiler.conda")
            set("Extracting Mojo…")
            try extractConda(compiler, into: enclaveMojoPrefix)
            let py = try await download(enclaveMojoPythonURL, name: "enclave-mojo-python.conda")
            try extractConda(py, into: enclaveMojoPrefix)
            recordMojoVersion(enclaveMojoPrefix, Self.enclaveMojoVersion)
        }
        try relocateMojoPrefix(enclaveMojoPrefix)

        // 2. enclave bundle — PREBUILT binary + runtime files (sandbox profiles,
        //    resources, web/dist) + prebuilt FFI shims, published by CI. No `.mojo`
        //    source; the binary was built + rpath-relocated + ad-hoc signed in CI by
        //    enclave/scripts/package_enclave.sh (mirrors the vault `millfolio`
        //    binary).
        try await ensureBundle()
        guard fm.fileExists(atPath: enclaveBin.path) else {
            throw BootstrapError.step("unpack", "enclave zip missing prebuilt enclave/build/enclave")
        }

        // 3. The binary is already built — nothing to compile on-device. Just ensure
        //    it's executable. (It links the Mojo runtime dylibs from mojo/lib via its
        //    relocated @loader_path rpath, and its per-query codegen still shells
        //    `mojo build` against the millfolio pkgs at runtime — the toolchain above
        //    stays for both.) The vault web UI is served by the app server; enclave
        //    here is only the harness/sandbox the generated programs run under.
        set("Installing enclave…")
        try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: enclaveBin.path)

        // 4. Put the bundle's FFI shims under the toolchain's lib/, so flare finds
        //    them via $CONDA_PREFIX/lib at runtime — enclave runs WITH CONDA_PREFIX
        //    set (it shells `mojo build` for the sandboxed generated-code compile),
        //    unlike the always-serving server.
        try installEnclaveShims()
        ensureConfig(at: enclaveConfigURL, Self.enclaveConfigDefault)
        recordStep(".enclave-step")
    }

    /// Copy the bundled relocatable FFI shims (+ their dylib deps) into the enclave
    /// Mojo prefix's lib/, where flare's `$CONDA_PREFIX/lib` lookup finds them.
    private func installEnclaveShims() throws {
        let fm = FileManager.default
        let libDir = enclaveMojoPrefix.appendingPathComponent("lib", isDirectory: true)
        try fm.createDirectory(at: libDir, withIntermediateDirectories: true)
        let buildDir = enclaveDir.appendingPathComponent("build", isDirectory: true)
        for name in (try? fm.contentsOfDirectory(atPath: buildDir.path)) ?? []
        where name.hasSuffix(".so") || name.hasSuffix(".dylib") {
            let dst = libDir.appendingPathComponent(name)
            try? fm.removeItem(at: dst)
            try fm.copyItem(at: buildDir.appendingPathComponent(name), to: dst)
        }
    }

    /// `mojo build` env for the enclave toolchain prefix.
    private func enclaveMojoEnv(python: URL) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let extraPath = "\(python.deletingLastPathComponent().path):\(enclaveMojoPrefix.appendingPathComponent("bin").path)"
        env["PATH"] = extraPath + ":" + (env["PATH"] ?? "/usr/bin:/bin")
        env["CONDA_PREFIX"] = enclaveMojoPrefix.path
        env["MODULAR_HOME"] = enclaveMojoPrefix.appendingPathComponent("share/max").path
        return env
    }

    // ── enclave: start (open a ready-to-use Terminal) ──────────────────────────
    /// enclave is a one-shot CLI, so "start" opens a Terminal in the install dir
    /// with the toolchain env pre-set — the user sets ANTHROPIC_API_KEY, points it
    /// at their data, and runs `./build/enclave`.
    public func startEnclave() {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.launchEnclaveTerminal() }
            catch { await self.set(failed: "enclave: \(humanError(error))") }
        }
    }

    /// Write the `run-enclave.sh` launcher — sets the toolchain env (enclave
    /// shells `mojo build` for the sandboxed generated-code compile), cd's to the
    /// install dir, and execs the enclave binary, forwarding any args (`"$@"`) as
    /// the task. Shared by the menu app (runs it in a NEW Terminal) and the CLI
    /// (execs it in the CURRENT terminal so enclave takes over stdin/stdout — a
    /// one-shot run with a task, or an interactive REPL with none). Returns its path.
    @discardableResult
    public func writeEnclaveScript() throws -> URL {
        let mojoBin = enclaveMojoPrefix.appendingPathComponent("bin").path
        let modularHome = enclaveMojoPrefix.appendingPathComponent("share/max").path
        // Single-quote paths (they live under "Application Support" — note the space).
        let script = support.appendingPathComponent("run-enclave.sh")
        let body = """
        #!/bin/bash
        cd '\(enclaveDir.path)'
        # Resolve sandbox/*.sb.template + resources/ by ABSOLUTE path (not cwd).
        export ENCLAVE_HOME='\(enclaveDir.path)'
        export CONDA_PREFIX='\(enclaveMojoPrefix.path)'
        export MODULAR_HOME='\(modularHome)'
        export PATH='\(mojoBin)':"$PATH"
        # The vault path shells `<millfolio>/build/mill manifest`, compiles the
        # generated program with `-I <millfolio>/src` + its vendored siblings, and
        # reads the ~/.config/mill index. enclave defaults to the dev sibling
        # layout (../millfolio); point it at the installed millfolio checkout instead.
        export ENCLAVE_MILLFOLIO='\(millfolioDir.path)'
        # flare's bundled OpenSSL has a CI-baked CA path; point it at the system
        # bundle so HTTPS to the Anthropic API verifies (else CertificateUntrusted).
        [ -f /etc/ssl/cert.pem ] && export SSL_CERT_FILE='/etc/ssl/cert.pem'
        exec ./build/enclave "$@"
        """
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script
    }

    public func launchEnclaveTerminal() async throws {
        let script = try writeEnclaveScript()
        let cmd = "'\(script.path)'"
        try run("/usr/bin/osascript",
                ["-e", "tell application \"Terminal\" to activate",
                 "-e", "tell application \"Terminal\" to do script \"\(cmd)\""])
    }

    // ── millfolio: install ────────────────────────────────────────────────────────
    /// Menu-app entry point: fire-and-forget, drives `phase`.
    public func installMillfolio() {
        guard !isBusy else { return }
        phase = .running("Starting…")
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.installMillfolioEngine(); await self.set(done: true) }
            catch { await self.set(failed: humanError(error)) }
        }
    }

    /// Download millfolio's Mojo toolchain + PRECOMPILED bundle and place it. Same
    /// nightly as enclave; the bundle ships a prebuilt binary + pkgs/*.mojoc
    /// (the vault surface + libs) + prebuilt FFI shims — no source, no on-device
    /// build. The toolchain is still needed to compile the per-query generated
    /// programs against `-I pkgs`, and to install the shims into its lib/.
    public func installMillfolioEngine() async throws {
        // Idempotent: skip the whole download if the binary is already there.
        // Granular: the vault binary runs WITH CONDA_PREFIX and dlopens its FFI
        // shims from `mojo/lib`, so check those exist too — a shared-toolchain
        // re-provision wipes them (the post-v0.4.35 `mill index` dlopen crash).
        let millfolioCritical = [
            millfolioBin,
            millfolioMojoPrefix.appendingPathComponent("lib/liblancedbmojo.dylib"),
            millfolioMojoPrefix.appendingPathComponent("lib/libzlibmojo.so"),
            millfolioDir.appendingPathComponent("pkgs/vault.mojoc"),
        ]
        if stepCurrent(".millfolio-step", millfolioCritical)
            && !mojoToolchainStale(millfolioMojoPrefix, Self.enclaveMojoVersion) {
            set("millfolio already installed — skipping")
            return
        }
        let fm = FileManager.default
        for d in [support, millfolioMojoPrefix, millfolioRoot, cacheDir] {
            try fm.createDirectory(at: d, withIntermediateDirectories: true)
        }
        logHeader("Install millfolio")

        // 1. Mojo toolchain (same nightly as enclave).
        if mojoToolchainStale(millfolioMojoPrefix, Self.enclaveMojoVersion) {
            set("Downloading Mojo compiler for millfolio (~70 MB)…")
            try? fm.removeItem(at: millfolioMojoPrefix)   // clear any stale nightly
            try fm.createDirectory(at: millfolioMojoPrefix, withIntermediateDirectories: true)
            let compiler = try await download(millfolioMojoCompilerURL, name: "millfolio-mojo-compiler.conda")
            set("Extracting Mojo…")
            try extractConda(compiler, into: millfolioMojoPrefix)
            let py = try await download(millfolioMojoPythonURL, name: "millfolio-mojo-python.conda")
            try extractConda(py, into: millfolioMojoPrefix)
            recordMojoVersion(millfolioMojoPrefix, Self.enclaveMojoVersion)
        }
        try relocateMojoPrefix(millfolioMojoPrefix)

        // 2. millfolio bundle — PRECOMPILED, no source. Ships pkgs/*.mojoc (the
        //    vault tool surface + its libs) + a prebuilt build/millfolio binary +
        //    the FFI shims. Commercial IP protection: no `.mojo` for the vault
        //    surface or its libs reaches the device, and there is no on-device
        //    source build.
        try await ensureBundle()
        guard fm.fileExists(atPath: millfolioBin.path) else {
            throw BootstrapError.step("unpack", "millfolio zip missing prebuilt millfolio/build/millfolio")
        }
        guard fm.fileExists(atPath: millfolioDir.appendingPathComponent("pkgs/vault.mojoc").path) else {
            throw BootstrapError.step("unpack", "millfolio zip missing precompiled millfolio/pkgs/vault.mojoc")
        }

        // 3. The binary is already built (shipped in build/millfolio) — nothing to
        //    compile on-device. Just ensure it's executable.
        set("Installing millfolio…")
        try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: millfolioBin.path)

        // 4. Put the bundle's FFI shims (libzlibmojo / liblancedbmojo / libflare_*
        //    + their dylib deps) under the toolchain's lib/, where each binding's
        //    `$CONDA_PREFIX/lib` lookup finds them at runtime (millfolio runs WITH
        //    CONDA_PREFIX set via run-millfolio.sh).
        try installMillfolioShims()
        recordStep(".millfolio-step")
    }

    /// Copy the bundled relocatable FFI shims (+ their dylib deps) into the millfolio
    /// Mojo prefix's lib/, where flare/zlib/lancedb's `$CONDA_PREFIX/lib` lookup
    /// finds them. Mirrors installEnclaveShims.
    private func installMillfolioShims() throws {
        let fm = FileManager.default
        let libDir = millfolioMojoPrefix.appendingPathComponent("lib", isDirectory: true)
        try fm.createDirectory(at: libDir, withIntermediateDirectories: true)
        let buildDir = millfolioDir.appendingPathComponent("build", isDirectory: true)
        for name in (try? fm.contentsOfDirectory(atPath: buildDir.path)) ?? []
        where name.hasSuffix(".so") || name.hasSuffix(".dylib") {
            let dst = libDir.appendingPathComponent(name)
            try? fm.removeItem(at: dst)
            try fm.copyItem(at: buildDir.appendingPathComponent(name), to: dst)
        }
    }

    /// The vault program enclave compiles + runs executes under enclave's
    /// CONDA_PREFIX (enclave-mojo), so the millfolio vault FFI shims it dlopens
    /// (liblancedbmojo / libzlibmojo + their dylib deps) must live in
    /// enclave-mojo/lib too. Copy the ones enclave lacks from the millfolio
    /// toolchain (same Mojo nightly → ABI-compatible). Best-effort; idempotent.
    public func linkVaultShims() {
        let fm = FileManager.default
        let src = millfolioMojoPrefix.appendingPathComponent("lib")
        let dst = enclaveMojoPrefix.appendingPathComponent("lib")
        guard fm.fileExists(atPath: src.path) else { return }
        try? fm.createDirectory(at: dst, withIntermediateDirectories: true)
        for name in (try? fm.contentsOfDirectory(atPath: src.path)) ?? []
        where name.hasSuffix(".dylib") || name.hasSuffix(".so") {
            let d = dst.appendingPathComponent(name)
            if !fm.fileExists(atPath: d.path) {   // don't clobber enclave's own shims
                try? fm.copyItem(at: src.appendingPathComponent(name), to: d)
            }
        }
    }

    /// Prime enclave's Mojo build cache so the FIRST vault query is warm.
    ///
    /// Every vault question compiles a ~20-line `from vault import *` program in
    /// the sandbox (enclave's CONDA_PREFIX). Cold, that recompiles the whole
    /// tool surface + its deps; warm (cache populated), it's a fraction of a
    /// second. Without this, the first query pays the full cold cost. We compile a
    /// throwaway program with the EXACT include set the harness uses
    /// (vaultcfg.vault_include_paths → millfolio/src + the vendored siblings),
    /// under enclave's toolchain env, which fills enclave-mojo's
    /// .mojo_cache. Best-effort + idempotent: a failure here just means the first
    /// real query warms it instead (no install failure).
    public func primeVaultCompile() {
        let fm = FileManager.default
        guard isEnclaveInstalled, isMillfolioInstalled,
              let python = try? findPython() else { return }
        let mojo = enclaveMojoPrefix.appendingPathComponent("bin/mojo").path
        // The harness's include set (mirror of vaultcfg.vault_include_paths):
        // the single millfolio/pkgs dir of precompiled `.mojoc`s (vault + its
        // libs). No source on the include path.
        let inc = [
            "-I", millfolioDir.appendingPathComponent("pkgs").path,
        ]
        let tmp = fm.temporaryDirectory.appendingPathComponent("millfolio-prime-\(UUID().uuidString)", isDirectory: true)
        defer { try? fm.removeItem(at: tmp) }
        do {
            try fm.createDirectory(at: tmp, withIntermediateDirectories: true)
            let gen = tmp.appendingPathComponent("gen.mojo")
            try """
            from vault import *
            def main() raises:
                print_answer("primed " + String(len(manifest())))
            """.write(to: gen, atomically: true, encoding: .utf8)
            set("Warming the vault compile cache…")
            // Same toolchain env the sandboxed per-query compile uses. Not
            // sandboxed here (trusted, our own stub), but it fills the SAME
            // enclave-mojo/.mojo_cache the sandboxed compile reads.
            try run(mojo, ["build", gen.path] + inc + ["-o", tmp.appendingPathComponent("gen").path],
                    cwd: enclaveDir, env: enclaveMojoEnv(python: python))
        } catch {
            // Best-effort: log and move on — the first real query will warm it.
            set("Vault compile-cache prime skipped (\(humanError(error)))")
        }
    }

    /// A fresh per-ask transcript path under the owner-only app-support tree:
    /// ~/Library/Application Support/Millfolio/sessions/<timestamp>-<slug>.log.
    /// These transcripts contain the REAL de-aliased answer (amounts, document
    /// facts), so they must not live in world-readable /tmp. The dir is created
    /// 0700 (owner-only), which blocks other local users from traversing to the
    /// logs regardless of each file's own mode.
    public func newSessionLog(for question: String) -> URL {
        let dir = support.appendingPathComponent("sessions", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
        // Tighten even if the directory already existed at a looser mode.
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o700], ofItemAtPath: dir.path)
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"
        let stamp = f.string(from: Date())
        var slug = ""
        for c in question.lowercased() {
            slug.append((c.isLetter || c.isNumber) ? c : "-")
            if slug.count >= 40 { break }
        }
        while slug.contains("--") { slug = slug.replacingOccurrences(of: "--", with: "-") }
        slug = slug.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        if slug.isEmpty { slug = "ask" }
        return dir.appendingPathComponent("\(stamp)-\(slug).log")
    }

    /// `mojo build` env for the millfolio toolchain prefix.
    private func millfolioMojoEnv(python: URL) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        let extraPath = "\(python.deletingLastPathComponent().path):\(millfolioMojoPrefix.appendingPathComponent("bin").path)"
        env["PATH"] = extraPath + ":" + (env["PATH"] ?? "/usr/bin:/bin")
        env["CONDA_PREFIX"] = millfolioMojoPrefix.path
        env["MODULAR_HOME"] = millfolioMojoPrefix.appendingPathComponent("share/max").path
        return env
    }

    /// Place the millfolio app bundle (millfolio/app). The app server (UI + REST +
    /// the streaming chat WS on one port) is shipped PREBUILT — built + rpath-
    /// relocated + ad-hoc signed in CI by app/scripts/package-app.sh (mirrors the
    /// vault `millfolio` binary) — so there's no on-device `mojo build`. It still
    /// runs under enclave's toolchain env at runtime (CONDA_PREFIX + the flare
    /// shims from mojo/lib), so it requires the enclave engine.
    public func installAppServer() async throws {
        if stepCurrent(".appserver-step", [appServerBin]) {
            set("millfolio app server already installed — skipping")
            return
        }
        guard isEnclaveInstalled else {
            throw BootstrapError.step("app server",
                "enclave engine not installed — run `mill install` first")
        }
        let fm = FileManager.default
        try fm.createDirectory(at: appRoot, withIntermediateDirectories: true)
        logHeader("Install millfolio app server")

        try await ensureBundle()
        guard fm.fileExists(atPath: appServerBin.path) else {
            throw BootstrapError.step("unpack", "millfolio-app.zip missing prebuilt build/millfolio-server")
        }

        // The binary is already built — nothing to compile on-device. Just ensure
        // it's executable. It links the Mojo runtime dylibs from mojo/lib via its
        // relocated @loader_path rpath, and at run time the app-server LaunchAgent
        // sets CONDA_PREFIX so flare's shims resolve from mojo/lib.
        set("Installing millfolio app server…")
        try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: appServerBin.path)
        recordStep(".appserver-step")
    }

    // ── millfolio: start (open a ready-to-use Terminal) ───────────────────────────
    /// millfolio is a one-shot vault CLI, so "start" opens a Terminal in the install
    /// dir with the toolchain env pre-set — the user runs e.g.
    /// `./build/mill manifest ~/.config/millfolio/vault`.
    public func startMillfolio() {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.launchMillfolioTerminal() }
            catch { await self.set(failed: "millfolio: \(humanError(error))") }
        }
    }

    /// Write the `run-millfolio.sh` launcher — sets the toolchain env, cd's to the
    /// install dir, and execs the millfolio binary forwarding any args (`"$@"`).
    /// Shared by the menu app (new Terminal) and the CLI (execs in the current
    /// terminal). Returns its path.
    @discardableResult
    public func writeMillfolioScript() throws -> URL {
        let mojoBin = millfolioMojoPrefix.appendingPathComponent("bin").path
        let modularHome = millfolioMojoPrefix.appendingPathComponent("share/max").path
        let script = support.appendingPathComponent("run-millfolio.sh")
        let body = """
        #!/bin/bash
        cd '\(millfolioDir.path)'
        export CONDA_PREFIX='\(millfolioMojoPrefix.path)'
        export MODULAR_HOME='\(modularHome)'
        export PATH='\(mojoBin)':"$PATH"
        exec ./build/millfolio "$@"
        """
        try body.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script
    }

    public func launchMillfolioTerminal() async throws {
        let script = try writeMillfolioScript()
        let cmd = "'\(script.path)'"
        try run("/usr/bin/osascript",
                ["-e", "tell application \"Terminal\" to activate",
                 "-e", "tell application \"Terminal\" to do script \"\(cmd)\""])
    }

    // ── millfolio: the VAULT umbrella (engine + enclave + vault) ──────────────────
    // millfolio is the umbrella entry point for the personal-data-vault use case. It
    // composes the three engines: the combined inference server (chat + embeddings
    // — both models' weights), enclave (the harness + its vault web chat), and the
    // millfolio vault tools/indexer.

    /// Resolve the vault dir: an explicit arg wins, then $MILLFOLIO_VAULT, then
    /// ~/.config/millfolio/vault. The Swift side always passes this through to the
    /// engines (MILLFOLIO_VAULT env / explicit arg), so it's the canonical location.
    public func vaultDir(_ arg: String? = nil) -> String {
        if let arg, !arg.isEmpty { return arg }
        let env = ProcessInfo.processInfo.environment["MILLFOLIO_VAULT"]
        if let env, !env.isEmpty { return env }
        return dotConfig.appendingPathComponent("millfolio/vault", isDirectory: true).path
    }

    /// Resolve the vault dir AND create it if missing. The millfolio binary's
    /// `manifest`/indexer require the vault dir to exist, but on a clean machine
    /// the default (~/.config/millfolio/vault) isn't there yet — so install/start would fail with
    /// "the directory … does not exist". Idempotent; returns the resolved path.
    @discardableResult
    public func ensureVaultDir(_ arg: String? = nil) -> String {
        let dir = vaultDir(arg)
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    /// `mill install` — install the combined inference server (+ both
    /// models' weights) + enclave + millfolio, idempotently. Each step skips what's
    /// already installed (see the guards in installServer/EnclaveEngine/Millfolio-
    /// Engine), so re-running is cheap and reuses anything present.
    // MARK: - Environment preflight (shared by `mill install` + `mill doctor`)

    /// One environment prerequisite: a human label, whether it passed, and the
    /// command that fixes it when it didn't.
    public struct EnvCheck: Sendable {
        public let label: String
        public let ok: Bool
        public let hint: String
    }

    /// Run a probe command, returning (trimmed combined output, exit code) and NEVER
    /// throwing — for env checks where a nonzero exit just means "not installed".
    private func probe(_ launch: String, _ args: [String]) -> (out: String, code: Int32) {
        guard FileManager.default.isExecutableFile(atPath: launch) else { return ("", 127) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launch)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        do { try p.run() } catch { return ("", 127) }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        let out = (String(data: data, encoding: .utf8) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (out, p.terminationStatus)
    }

    /// The BUILD prerequisites `mill install` needs, one EnvCheck each — so the install
    /// preflight and `mill doctor` share ONE source of truth. Pure inspection:
    ///   • Apple Silicon (arm64).
    ///   • FULL Xcode selected (not just Command Line Tools) — `xcodebuild
    ///     -downloadComponent` and the Metal compiler live in Xcode, not the CLT.
    ///   • Metal Toolchain present — the engine compiles Metal GPU kernels at build
    ///     time; without it `mojo build` fails mid-install with `Metal Compiler failed
    ///     to compile metallib`. Probed with `xcrun metal --version`, which exits
    ///     nonzero ("missing Metal Toolchain") when the component isn't downloaded —
    ///     unlike `xcrun --find metal`, whose wrapper can exist while it's missing.
    ///   • Python ≥ 3.10 (the Mojo toolchain needs one).
    public func preflightEnv() -> [EnvCheck] {
        var checks: [EnvCheck] = []

        let arch = probe("/usr/bin/uname", ["-m"]).out
        let osVer = probe("/usr/bin/sw_vers", ["-productVersion"]).out
        checks.append(EnvCheck(
            label: "Apple Silicon — \(arch.isEmpty ? "?" : arch), macOS \(osVer.isEmpty ? "?" : osVer)",
            ok: arch == "arm64",
            hint: "millfolio requires an Apple-Silicon Mac"))

        // Full Xcode: the active developer dir must live inside an Xcode.app, not the
        // CommandLineTools dir (which can't download the Metal Toolchain).
        let devDir = probe("/usr/bin/xcode-select", ["-p"]).out
        checks.append(EnvCheck(
            label: "Full Xcode — \(devDir.isEmpty ? "none selected" : devDir)",
            ok: devDir.contains(".app/Contents/Developer"),
            hint: "install Xcode (App Store), then: sudo xcode-select -s /Applications/Xcode.app"))

        // Metal Toolchain: succeeds only when the downloadable component is present.
        checks.append(EnvCheck(
            label: "Metal toolchain",
            ok: probe("/usr/bin/xcrun", ["metal", "--version"]).code == 0,
            hint: "install: xcodebuild -downloadComponent MetalToolchain"))

        // Python ≥ 3.10.
        var pyLabel = "not found"
        var pyOK = false
        if let py = ["/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"]
            .first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
            let v = probe(py, ["--version"]).out  // e.g. "Python 3.12.4"
            pyLabel = v.isEmpty ? py : v
            let parts = v.replacingOccurrences(of: "Python ", with: "").split(separator: ".")
            if parts.count >= 2, let major = Int(parts[0]), let minor = Int(parts[1]) {
                pyOK = major > 3 || (major == 3 && minor >= 10)
            }
        }
        checks.append(EnvCheck(label: "Python 3 — \(pyLabel)", ok: pyOK, hint: "Python ≥ 3.10 is required"))

        return checks
    }

    /// Fail FAST — before any build — if a build prerequisite is missing, so e.g. a
    /// missing Metal Toolchain surfaces immediately as "run X, then re-run mill install"
    /// rather than as a cryptic `metallib` compile error several steps in. `mill doctor`
    /// shows the same checks without throwing.
    public func requireEnv() throws {
        let failed = preflightEnv().filter { !$0.ok }
        guard failed.isEmpty else {
            let lines = failed.map { "  ✗ \($0.label)\n      → \($0.hint)" }.joined(separator: "\n")
            throw BootstrapError.preflight(
                "millfolio can't install — missing prerequisite(s):\n\(lines)\n\n"
                + "Fix the above, then re-run `mill install`  (`mill doctor` re-checks).")
        }
    }

    public func installVault() async throws {
        try requireEnv()                    // fail fast on missing Xcode/Metal/Python
        try await installServer()           // engine + chat + embedding weights
        // Millfolio (vault tools + precompiled pkgs) BEFORE enclave: the
        // harness now builds against the vault pkg for in-process tags, so the
        // pkgs must be unpacked first.
        try await installMillfolioEngine()    // the vault tools + indexer + pkgs
        try await installEnclaveEngine()   // the vault harness/sandbox (needs vault pkgs)
        try await installAppServer()        // the millfolio web app (UI on :10000, WS on :10001)
        linkVaultShims()                    // millfolio FFI shims → enclave-mojo/lib (vault-run dlopen)
        primeVaultCompile()                 // warm enclave-mojo's .mojo_cache so query #1 is fast
        ensureVaultDir()                    // leave the default vault dir ready
    }

    /// Menu-app entry point: fire-and-forget umbrella install, drives `phase`.
    public func installVaultFireAndForget() {
        guard !isBusy else { return }
        phase = .running("Starting…")
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.installVault(); await self.set(done: true) }
            catch { await self.set(failed: humanError(error)) }
        }
    }

    /// The app server (UI + REST + chat WS on :10000) as a launchd agent — the SAME
    /// mechanism as the inference server (no more nohup orphan / pkill race). Runs the
    /// millfolio-server binary from enclave's dir (so `sandbox/*.sb.template`
    /// resolve), the UI from MILLFOLIO_WEB_DIR (absolute), with the toolchain env
    /// (CONDA_PREFIX + flare shims) + the vault-resolution env. Returns the plist URL.
    private func writeAppServerLaunchAgent(vaultDir dir: String) throws -> URL {
        try FileManager.default.createDirectory(
            at: appServerLaunchAgentURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: millfolioLogDir, withIntermediateDirectories: true)
        let mojoBin = enclaveMojoPrefix.appendingPathComponent("bin").path
        // The engine runner the app server shells for /api/search (LanceDB stays out
        // of the web server). Ensure it exists + hand the server its path.
        let runScript = try writeMillfolioScript()
        var env: [String: String] = [
            "CONDA_PREFIX": enclaveMojoPrefix.path,
            "MODULAR_HOME": enclaveMojoPrefix.appendingPathComponent("share/max").path,
            // launchd doesn't inherit a login PATH — give it mojo's bin + system dirs.
            "PATH": "\(mojoBin):/usr/bin:/bin:/usr/sbin:/sbin",
            "ENCLAVE_VAULT_DIR": dir,
            "MILLFOLIO_VAULT": dir,
            // The vault tools (search/ask_local) hit the inference server over loopback.
            "MILLFOLIO_EMBED_URL": "http://127.0.0.1:8000/v1",
            "MILLFOLIO_LOCAL_URL": "http://127.0.0.1:8000/v1",
            // Weight provisioning at RUNTIME (moved out of the installer): the app
            // server runs this native-Mojo downloader to fetch the embedding model +
            // a default chat model in the background, and to fulfil the UI catalog's
            // on-demand downloads. HF_HOME is where they land (the self-contained
            // cache the engine reads); set it explicitly now that downloads run here.
            "MILLFOLIO_DOWNLOAD_BIN": downloadBin.path,
            "HF_HOME": hfHome.path,
            // The chat WS compiles the generated program against the millfolio sources.
            "ENCLAVE_MILLFOLIO": millfolioDir.path,
            // The enclave install dir — so it resolves its sandbox/*.sb.template
            // profiles + resources/enclave-system.md by ABSOLUTE path, not the
            // process cwd (WorkingDirectory below is belt-and-suspenders).
            "ENCLAVE_HOME": enclaveDir.path,
            // Serve the built UI by ABSOLUTE path so it doesn't depend on cwd.
            "MILLFOLIO_WEB_DIR": appRoot.appendingPathComponent("web/dist").path,
            "MILLFOLIO_RUN_SCRIPT": runScript.path,
            // A few worker threads so the long synchronous chat handler (codegen →
            // compile → run, ~30s+) doesn't monopolize a single-threaded reactor and
            // freeze the rest of the UI (Stats/System/Vault GETs would block on it).
            // Safe: the sandboxed RUN stays serial via the flock run-queue regardless.
            "MILLFOLIO_WORKERS": "4",
            // Surface the on-device-model exchanges (ask_local/ask_local_batch sent →
            // got) as collapsible debug items in the chat — for debugging an answer
            // (e.g. why a phone-bill filter matched what it did).
            "MILLFOLIO_LOG_LOCAL": "1",
        ]
        if FileManager.default.fileExists(atPath: "/etc/ssl/cert.pem") {
            env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
        }
        // The release version — surfaced as the running version in /api/system + the
        // UI build stamp. Prefer the version stamped INTO the bundle at build time
        // (bundleRoot/VERSION, e.g. "0.4.44-rc.4"): correct for the Mac-app install
        // path (which has no brew, so brewCliVersion() is empty → server showed "dev").
        // Fall back to the brew CLI version for older bundles lacking the file; leave
        // unset for an unmanaged source build (then the server reports "dev").
        let stampedVersion = (try? String(contentsOf: bundleRoot.appendingPathComponent("VERSION"),
                                          encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let resolvedVersion = stampedVersion.isEmpty ? brewCliVersion() : stampedVersion
        if !resolvedVersion.isEmpty {
            env["MILLFOLIO_VERSION"] = resolvedVersion
        }
        // Forward the frontier-model credentials from the invoking shell into the
        // daemon. launchd agents DON'T inherit the login shell, so without this the
        // app server sees no ANTHROPIC_API_KEY → settings.load_config sets
        // token_budget=0 → EVERY codegen falls back to the weak local model (which
        // fabricates + truncates the program). The privacy design wants the frontier
        // model WRITING programs (it sees only the aliased manifest, never data), so
        // forward the key (+ optional base-url / budget overrides) when present.
        // NOTE: this persists the key in the launch-agent plist (plaintext, under
        // ~/Library/LaunchAgents). We chmod the plist 0600 below so it is
        // owner-only on disk (PropertyListSerialization.write would otherwise
        // leave it group/other-readable at the umask default). To avoid the
        // on-disk key entirely, put it in ~/.config/enclave/config.json
        // (0600) instead — settings reads either.
        for key in ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ENCLAVE_REMOTE_TOKEN_BUDGET"] {
            if let v = ProcessInfo.processInfo.environment[key], !v.isEmpty {
                env[key] = v
            }
        }
        let serverLog = millfolioLogDir.appendingPathComponent("server.log").path
        // Run the server through a millfolio-NAMED wrapper script rather than a bare
        // `/bin/sh -c …` purely for NAMING: macOS's "… can run in the background"
        // notification (and the Login Items list) names the item after
        // ProgramArguments[0]'s basename, so a script named `millfolio-appserver`
        // brands it as ours (a bare interpreter shows the generic "sh"). The server's
        // own logger (logging.mojo) stamps every line with a full local
        // `[YYYY-MM-DD HH:MM:SS.mmm]` at the moment it is written, so there is no
        // post-hoc timestamping filter — the wrapper just execs the server with its
        // output appended to the log (exec so signals reach the server directly).
        let wrapper = support.appendingPathComponent("millfolio-appserver")
        let wrapperBody = """
        #!/bin/sh
        # millfolio app server (launchd background item).
        exec "\(appServerBin.path)" >> "\(serverLog)" 2>&1
        """
        try wrapperBody.write(to: wrapper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: wrapper.path)
        let plist: [String: Any] = [
            "Label": Self.appServerLabel,
            "ProgramArguments": [wrapper.path],
            "WorkingDirectory": enclaveDir.path,   // sandbox/*.sb.template resolve here
            "EnvironmentVariables": env,
            "StandardErrorPath": serverLog,            // pre-exec wrapper (sh) errors only
            "RunAtLoad": true,
            "ProcessType": "Interactive",
        ]
        let data = try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
        try data.write(to: appServerLaunchAgentURL)
        // The plist embeds ANTHROPIC_API_KEY in cleartext; keep it owner-only so
        // it isn't exposed to other local users, Spotlight, or a home-dir perm slip.
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: appServerLaunchAgentURL.path)
        return appServerLaunchAgentURL
    }

    /// Self-heal the vault FFI shims: the app server runs generated `from vault
    /// import *` programs that `dlopen` liblancedbmojo.dylib / libzlibmojo.so from
    /// $CONDA_PREFIX/lib (= the toolchain prefix). A full install copies them
    /// (installMillfolioShims), but a present-binary-but-missing-shim state (e.g. a
    /// partial/older install) would otherwise crash search() at runtime with a
    /// dlopen error. Re-run the (idempotent) shim copy if the marquee shim is
    /// missing, so `mill start` always leaves them in place.
    private func ensureVaultShims() {
        let lance = mojoPrefix.appendingPathComponent("lib/liblancedbmojo.dylib")
        let bundled = millfolioDir.appendingPathComponent("build/liblancedbmojo.dylib")
        if !FileManager.default.fileExists(atPath: lance.path)
            && FileManager.default.fileExists(atPath: bundled.path) {
            try? installMillfolioShims()
        }
    }

    /// Start the app server under launchd, wait until :10000 answers, then expose it
    /// on the tailnet + open the browser (one-shot — not part of the daemon).
    ///
    /// `openBrowser` defaults to true (the CLI's behavior — unchanged). The menu-bar
    /// app passes `false`: it IS the browser (the native WKWebView `WebWindow`), so
    /// it must NOT also `open` the system default browser or `tailscale serve` — it
    /// just needs :10000 up. The wait-for-ready is done either way.
    public func startAppServer(vaultDir dir: String, openBrowser: Bool = true) async throws {
        ensureVaultShims()   // guard search()'s dlopen before the server runs programs
        let url = try writeAppServerLaunchAgent(vaultDir: dir)
        _ = await runStatusAsync("/bin/launchctl", ["bootout", "\(guiDomain)/\(Self.appServerLabel)"])
        await killStaleOnPort(10000)
        await killStaleOnPort(10001)   // reap any legacy nohup instance
        try await runAsync("/bin/launchctl", ["bootstrap", guiDomain, url.path])
        _ = await waitForPort(10000, timeout: 25)
        // Don't open the browser on "socket listening" alone — the multi-worker app
        // server BINDS the port before its workers can serve files, so the first load
        // raced the `_app/*.js` chunks to 404 (a page refresh fixed it). Wait until a
        // real STATIC asset (the favicon, served from MILLFOLIO_WEB_DIR) returns 200 —
        // then the JS chunks are being served too. Falls through + opens on timeout.
        _ = await waitForHttp(10000, path: "/favicon.svg", timeout: 25)
        guard openBrowser else { return }
        _ = await runStatusAsync("/bin/bash", ["-c",
            "command -v tailscale >/dev/null 2>&1 && tailscale serve --bg 10000 >/dev/null 2>&1 || true"])
        _ = await runStatusAsync("/bin/bash", ["-c", "open 'http://localhost:10000' >/dev/null 2>&1 &"])
    }

    /// Poll until something is LISTENING on `port`, or `timeout` s elapse.
    /// Async poll — suspends between probes instead of sleeping the main thread.
    @discardableResult
    public func waitForPort(_ port: Int, timeout: Double = 25) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await runStatusAsync("/bin/bash", ["-c",
                "lsof -ti tcp:\(port) -sTCP:LISTEN >/dev/null 2>&1"]) == 0 { return true }
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        return false
    }

    /// Poll until an HTTP GET of `path` on `port` returns 200, or `timeout` s elapse.
    /// Stronger than `waitForPort`: it confirms the server is actually SERVING (the
    /// socket can be listening before the workers serve content). Probe a real static
    /// asset to confirm the web root is mounted + serving.
    @discardableResult
    public func waitForHttp(_ port: Int, path: String = "/", timeout: Double = 25) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await Self.httpGet("http://localhost:\(port)\(path)")?.code == 200 { return true }
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        return false
    }

    /// GET `url` with a short timeout — Foundation URLSession, NOT a shelled
    /// `curl` (no subprocess, structured errors, and inherently off-main). nil on
    /// any transport failure (connection refused = "not serving yet").
    nonisolated private static func httpGet(_ url: String, timeout: Double = 2)
        async -> (code: Int, body: String)?
    {
        guard let u = URL(string: url) else { return nil }
        var req = URLRequest(url: u)
        req.timeoutInterval = timeout
        req.cachePolicy = .reloadIgnoringLocalCacheData
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              let http = resp as? HTTPURLResponse else { return nil }
        return (http.statusCode, String(data: data, encoding: .utf8) ?? "")
    }

    /// `mill start`: ensure the combined inference server is running (launchd),
    /// then start the vault app servers in the BACKGROUND (no Terminal) and open
    /// http://localhost:10000. Server output goes to the millfolio server log.
    public func startVaultChat(vaultDir dir: String, openBrowser: Bool = true) async throws {
        // 0. The vault dir must exist before enclave/millfolio's `manifest` runs
        //    over it (a clean machine has no vault dir yet).
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        // 1. Kick the engine agent WITHOUT waiting — it binds and serves /v1/status
        //    instantly and loads the model in its own background thread.
        let engineComing = try await kickInferenceServer()
        // 2. App servers up FIRST (they start in ~a second), so the app's WKWebView
        //    has something to render while the model loads instead of the whole
        //    start hanging on the engine. Stop any prior agent + reap stale ports,
        //    build it if an older install predates it (self-heal on upgrade), then
        //    (re)start it under launchd.
        _ = await stopAppServer()
        if !isAppServerInstalled { try await installAppServer() }
        // `openBrowser` forwarded: the menu-bar app passes false (it renders :10000 in
        // its own WKWebView, so it must not also spawn the system browser).
        try await startAppServer(vaultDir: dir, openBrowser: openBrowser)
        // 3. Now surface the engine's load progress (phases via /v1/status) in the
        //    app instead of hanging silently; this throws only when the engine is
        //    dead or reports an error (e.g. model not downloaded), not when slow.
        if engineComing {
            try await waitForEngineReady()
            serverRunning = true
        }
    }

    /// Ensure the combined inference server is running AND ready (idempotent).
    /// No-op if it isn't installed yet — the caller surfaces that downstream.
    /// Without this, `ask`/the vault loop blocks on a dead model endpoint with no
    /// clue why. Readiness is awaited with progress via `waitForEngineReady`.
    public func ensureInferenceServer() async throws {
        guard try await kickInferenceServer() else { return }
        try await waitForEngineReady()
        serverRunning = true
    }

    /// Start the engine agent WITHOUT waiting for model readiness. Returns true
    /// when the engine is expected to become ready (already serving, or launched
    /// with weights present) — i.e. when a `waitForEngineReady` is worthwhile.
    @discardableResult
    public func kickInferenceServer() async throws -> Bool {
        guard isServerInstalled else { return false }
        // Probe the PORT, not just launchd state — a "loaded" agent can be dead or
        // still loading weights. If it's already serving, we're done.
        if await inferenceListening() { serverRunning = true; return true }
        await killStaleOnPort(8000)  // reap a half-dead instance holding the port
        // No chat weights yet (fresh install, before the app server's background
        // provisioner has fetched the default model)? Bootstrap the engine agent so
        // it's loaded + kickstartable — but DON'T wait for readiness (it can't serve
        // until the weights land; the provisioner kickstarts it then).
        guard weightsPresent else {
            set("Model weights aren't present yet — the app will download them in the background, then start the engine…")
            try? await startServer() // bootstrap the (config-authoritative) agent
            return false
        }
        set("Starting the inference server…")
        try await startServer()      // bootout + bootstrap (RunAtLoad)
        return true
    }

    /// Wait until the engine reports ready on /v1/status, surfacing load phases
    /// ("loading model weights…", "warming up GPU kernels…") through `set` so the
    /// app shows progress instead of hanging. Distinguishes:
    /// - slow but loading → keeps waiting (phase updates prove liveness),
    /// - an engine-reported error (e.g. "model not downloaded") → throws with it,
    /// - a dead engine (nothing listening on :8000) → throws quickly,
    /// - a pre-/v1/status engine (≤ v0.4.50-rc.4: silent while loading, 404 once
    ///   ready) → falls back to the legacy 90 s /v1/version wait.
    public func waitForEngineReady(maxWait: TimeInterval = 900) async throws {
        let deadline = Date().addingTimeInterval(maxWait)
        var lastPhase = ""
        var silentSince: Date?      // answering nothing at all since…
        while Date() < deadline {
            if let r = await Self.httpGet("http://127.0.0.1:8000/v1/status") {
                silentSince = nil
                if r.code == 200, let state = Self.jsonField("state", in: r.body) {
                    switch state {
                    case "ready":
                        return
                    case "error":
                        let msg = Self.jsonField("error", in: r.body) ?? "unknown engine error"
                        throw BootstrapError.step("start server",
                            "the inference engine reports: \(msg)")
                    default:        // "loading"
                        let phase = Self.jsonField("phase", in: r.body) ?? "loading"
                        if phase != lastPhase {
                            lastPhase = phase
                            set("Inference engine: \(phase)…")
                        }
                    }
                } else if r.code == 404 {
                    // An older engine only answers once READY (it 404s the unknown
                    // /v1/status route) — confirm via the legacy readiness probe.
                    if await waitForInference(timeout: 90) { return }
                    throw BootstrapError.step("start server",
                        "the inference server didn't become ready on :8000 within 90s — see \(logFileURL.path)")
                }
                // other codes: transitional — keep polling
            } else if await portListening(8000) {
                // Listening but not answering: an older engine mid-load queues
                // requests silently. No status to stream — say so once and wait
                // out the legacy load window.
                if lastPhase != "legacy" {
                    lastPhase = "legacy"
                    set("Inference engine is loading the model (older engine — no progress available)…")
                }
                if silentSince == nil { silentSince = Date() }
                if Date().timeIntervalSince(silentSince!) > 180 {
                    throw BootstrapError.step("start server",
                        "the inference server has been silent on :8000 for 3 minutes — see \(logFileURL.path)")
                }
            } else {
                // Nothing listening: the new engine binds instantly, so more than
                // a few seconds of this means dead, not loading (launchd may be
                // between respawns — allow a short grace).
                if silentSince == nil { silentSince = Date() }
                if Date().timeIntervalSince(silentSince!) > 15 {
                    throw BootstrapError.step("start server",
                        "the inference server isn't listening on :8000 — see \(logFileURL.path)")
                }
            }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
        throw BootstrapError.step("start server",
            "the inference server didn't become ready within \(Int(maxWait / 60)) minutes — see \(logFileURL.path)")
    }

    /// Minimal `"key":"value"` string extraction — enough for the engine's flat
    /// /v1/status body; avoids JSONSerialization for a hot 1 Hz poll.
    nonisolated static func jsonField(_ key: String, in body: String) -> String? {
        guard let k = body.range(of: "\"\(key)\":\"") else { return nil }
        guard let end = body.range(of: "\"", range: k.upperBound..<body.endIndex) else { return nil }
        return String(body[k.upperBound..<end.lowerBound])
    }

    /// Is anything LISTENING on `port`? (Distinguishes a dead engine from an
    /// older one that accepts but queues requests while loading weights.)
    private func portListening(_ port: Int) async -> Bool {
        await runStatusAsync("/bin/bash", ["-c",
            "lsof -ti tcp:\(port) -sTCP:LISTEN >/dev/null 2>&1"]) == 0
    }

    /// Stop the app server. Bootout the launchd agent (the current mechanism), and
    /// also pkill any legacy nohup-launched instance (pre-launchd installs) + reap
    /// the ports. Returns true if anything was running.
    public func stopAppServer() async -> Bool {
        let wasLoaded = await runStatusAsync("/bin/launchctl", ["print", "\(guiDomain)/\(Self.appServerLabel)"]) == 0
        _ = await runStatusAsync("/bin/launchctl", ["bootout", "\(guiDomain)/\(Self.appServerLabel)"])
        // Legacy nohup processes from an older install (no launchd agent): pkill them.
        let ws = await runStatusAsync("/usr/bin/pkill", ["-f", "build/millfolio-ws"]) == 0
        let srv = await runStatusAsync("/usr/bin/pkill", ["-f", "build/millfolio-server"]) == 0
        for port in [10000, 10001] { await killStaleOnPort(port) }
        return wasLoaded || ws || srv
    }

    // ── server readiness / port hygiene ──────────────────────────────────────
    /// A REAL listening check: does the inference server answer on :8000? (A
    /// launchd-"loaded" agent can still be mid-load or dead, so we probe the port,
    /// not just launchctl state.) Returns the reported build version, or nil if it
    /// isn't answering yet.
    public func inferenceVersion() async -> String? {
        guard let r = await Self.httpGet("http://127.0.0.1:8000/v1/version"),
              r.code == 200, case let out = r.body,
              out.contains("\"version\"") else { return nil }
        // Pull the version string out of {"engine":"millfolio","version":"…"}.
        guard let r = out.range(of: "\"version\"") ,
              let q1 = out.range(of: "\"", range: r.upperBound..<out.endIndex),
              let q2 = out.range(of: "\"", range: q1.upperBound..<out.endIndex)
        else { return "" }
        return String(out[q1.upperBound..<q2.lowerBound])
    }

    public func inferenceListening() async -> Bool { await inferenceVersion() != nil }

    /// Poll until the inference server answers on :8000, or `timeout` s elapse.
    /// Async poll — suspends between probes instead of sleeping the main thread.
    @discardableResult
    public func waitForInference(timeout: Double = 60) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await inferenceListening() { return true }
            try? await Task.sleep(nanoseconds: 400_000_000)
        }
        return await inferenceListening()
    }

    /// Reap any process LISTENING on `port` that launchd/pkill missed, so a fresh
    /// server binds without AddressInUse. Best-effort; never throws.
    public func killStaleOnPort(_ port: Int) async {
        _ = await runStatusAsync("/bin/bash", ["-c",
            "p=$(lsof -ti tcp:\(port) -sTCP:LISTEN 2>/dev/null); [ -n \"$p\" ] && kill $p 2>/dev/null; true"])
    }

    /// Menu-app entry point: open the vault chat (fire-and-forget).
    /// `openBrowser` defaults true (CLI behavior). The menu-bar app passes false — it
    /// renders :10000 in its own WKWebView and must not also spawn the system browser.
    public func startVaultChatFireAndForget(openBrowser: Bool = true) {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do { try await self.startVaultChat(vaultDir: self.vaultDir(), openBrowser: openBrowser) }
            catch { await self.set(failed: "vault chat: \(humanError(error))") }
        }
    }

    // ── standalone-app bundle auto-refresh ───────────────────────────────────────
    /// The bundle version recorded on disk (the `.bundle-version` stamp `ensureBundle`
    /// writes), or "" when no bundle is unpacked yet.
    private func installedBundleVersion() -> String {
        (try? String(contentsOf: bundleRoot.appendingPathComponent(".bundle-version"),
                     encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    /// **App only.** If a newer PROD `millfolio.zip` has shipped since this app last
    /// provisioned, re-provision against it (re-fetch the bundle + rebuild every
    /// on-device component); otherwise a no-op. The Mojo toolchain (`mojo/`) and the
    /// model weights (`hf/`) live OUTSIDE `bundleRoot`, so a refresh only re-fetches the
    /// source bundle and recompiles the on-device engine — weights are preserved.
    ///
    /// Offline-safe: when the prod-latest lookup fails we keep the installed bundle
    /// (never wipe it). Progress streams through `phase`/`onProgress`, so the caller can
    /// surface it. Runs the SAME `installVault` provision path used on first run.
    /// Returns true iff it actually re-provisioned to a newer bundle (so the caller
    /// knows whether the servers need (re)starting). No-op paths return false.
    @discardableResult
    public func refreshBundleIfStale() async throws -> Bool {
        guard forceLatestBundle else { return false }   // the CLI pins to its brew tag; no-op there
        guard isProvisioned else { return false }       // not provisioned → first-run onboarding handles it
        guard let latest = await resolveProdLatestVersion() else {
            set("Couldn't check for a newer millfolio release (offline?) — keeping the installed one.")
            return false
        }
        resolvedProdLatest = latest               // pin the freshness key + staleness `want`
        let have = installedBundleVersion()
        guard have != latest else { return false }      // already current
        set("Updating millfolio to \(latest)… (this can take a couple of minutes)")
        // Stop the app server so its binary can be replaced, drop the unpacked bundle so
        // `ensureBundle` re-fetches, then re-run the full provision. Because
        // `installVersionKey` now equals the NEW prod tag, every per-step guard re-fires
        // and rebuilds against the fresh source.
        _ = await stopAppServer()
        try? FileManager.default.removeItem(at: bundleRoot)
        try await installVault()
        return true
    }

    /// User-initiated "Check for Updates" bundle path (servers already running):
    /// refresh the prod bundle and RESTART onto it only if it actually changed — so an
    /// up-to-date check leaves the running servers alone (no gratuitous restart). Paired
    /// with the Sparkle app-update check so one button updates both the app and the bundle.
    public func refreshBundleAndRestartIfChangedFireAndForget(openBrowser: Bool = false) {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                let changed = try await self.refreshBundleIfStale()
                if changed {
                    try await self.startVaultChat(vaultDir: self.vaultDir(), openBrowser: openBrowser)
                }
                await self.set(done: true)
            } catch {
                await self.set(failed: "update: \(humanError(error))")
            }
        }
    }

    /// App-launch entry point (provisioned): refresh the bundle if a newer PROD release
    /// has shipped, then bring the local servers up — all in the background so the menu
    /// bar / UI never blocks. Up-to-date or offline → just starts the servers on the
    /// current bundle. `openBrowser:false` for the menu-bar app (it renders :10000 in
    /// its own WKWebView). On completion `phase` returns to `.done`.
    public func refreshBundleThenStartFireAndForget(openBrowser: Bool = false) {
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            do {
                try await self.refreshBundleIfStale()
                try await self.startVaultChat(vaultDir: self.vaultDir(), openBrowser: openBrowser)
                await self.set(done: true)
            } catch {
                await self.set(failed: "startup: \(humanError(error))")
            }
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────────
    @discardableResult
    private func run(_ launch: String, _ args: [String], cwd: URL? = nil, env: [String: String]? = nil) throws -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launch)
        p.arguments = args
        if let cwd { p.currentDirectoryURL = cwd }
        if let env { p.environment = env }
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        appendLog("\n$ \(launch) \(args.joined(separator: " "))\n")
        try p.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        let out = String(data: data, encoding: .utf8) ?? ""
        appendLog(out)
        if p.terminationStatus != 0 {
            appendLog("\n[\(URL(fileURLWithPath: launch).lastPathComponent) exited \(p.terminationStatus)]\n")
            throw BootstrapError.step(URL(fileURLWithPath: launch).lastPathComponent,
                                      "exit \(p.terminationStatus): " + out.suffix(500))
        }
        return out
    }

    /// Like `run`, but returns the exit status instead of throwing on nonzero —
    /// for probes (launchctl print/bootout) where a nonzero code is expected.
    @discardableResult
    private func runStatus(_ launch: String, _ args: [String]) throws -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launch)
        p.arguments = args
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        try p.run()
        _ = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return p.terminationStatus
    }

    // ── off-main subprocess primitives (the no-beachball path) ────────────────
    // `run`/`runStatus` above SPAWN + WAIT synchronously — called from this
    // @MainActor class that blocks the main thread (spinning beachball) for the
    // subprocess's whole lifetime. The app-launch path (start servers, port/HTTP
    // readiness polls, launchctl probes) goes through these async twins instead:
    // the blocking spawn/wait happens on a background queue and the @MainActor
    // caller just suspends, so the menu bar + windows stay responsive.

    /// Spawn + wait on a background queue. `spawnError` carries a failed exec
    /// (missing binary); otherwise combined output + exit status, like `run`.
    nonisolated private static func spawnProcess(
        _ launch: String, _ args: [String], cwd: URL? = nil, env: [String: String]? = nil
    ) async -> (out: String, code: Int32, spawnError: Error?) {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let p = Process()
                p.executableURL = URL(fileURLWithPath: launch)
                p.arguments = args
                if let cwd { p.currentDirectoryURL = cwd }
                if let env { p.environment = env }
                let pipe = Pipe()
                p.standardOutput = pipe
                p.standardError = pipe
                do { try p.run() } catch {
                    cont.resume(returning: ("", -1, error))
                    return
                }
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                p.waitUntilExit()
                cont.resume(returning: (String(data: data, encoding: .utf8) ?? "",
                                        p.terminationStatus, nil))
            }
        }
    }

    /// `run`, off-main: the same log + throw-on-nonzero contract, main actor free
    /// while the subprocess runs.
    @discardableResult
    private func runAsync(_ launch: String, _ args: [String],
                          cwd: URL? = nil, env: [String: String]? = nil) async throws -> String {
        appendLog("\n$ \(launch) \(args.joined(separator: " "))\n")
        let r = await Self.spawnProcess(launch, args, cwd: cwd, env: env)
        if let e = r.spawnError { throw e }
        appendLog(r.out)
        if r.code != 0 {
            appendLog("\n[\(URL(fileURLWithPath: launch).lastPathComponent) exited \(r.code)]\n")
            throw BootstrapError.step(URL(fileURLWithPath: launch).lastPathComponent,
                                      "exit \(r.code): " + r.out.suffix(500))
        }
        return r.out
    }

    /// `runStatus`, off-main — for probes where nonzero is expected. A failed
    /// exec maps to -1 (the probes only compare against 0).
    @discardableResult
    private func runStatusAsync(_ launch: String, _ args: [String]) async -> Int32 {
        let r = await Self.spawnProcess(launch, args)
        return r.spawnError == nil ? r.code : -1
    }

    // ── diagnosable one-shot runs (ask / index) ────────────────────────────────
    // The `ask` and `index` subcommands used to execv /bin/bash, which REPLACES
    // this process — so a failure inside the child (e.g. enclave's `posix_spawn`
    // of the mojo compiler failing with ENOENT) left nothing to log. These run the
    // launcher as a child instead, mirroring its combined stdout/stderr to both the
    // terminal and the millfolio log, after dumping the launcher + the paths it
    // depends on. Returns the child's exit status (caller maps it to the CLI exit).

    /// Run the enclave vault loop for one question. See runLoggedScript.
    public func runVaultAsk(question: String, vaultDir: String) throws -> Int32 {
        refreshServerRunning()
        ensureVaultShims()  // self-heal a wiped shared-toolchain lib/ before the vault binary dlopens it
        let script = try writeEnclaveScript()
        let args = ["vault", question, vaultDir]
        logRunDiagnostics(label: "ask", launcher: script, args: args, probes: [
            ("enclave launcher", script.path),
            ("enclave dir (cwd)", enclaveDir.path),
            ("enclave binary", enclaveBin.path),
            ("mojo compiler (enclave shells it)", enclaveMojoPrefix.appendingPathComponent("bin/mojo").path),
            ("millfolio vault tools (src)", millfolioDir.appendingPathComponent("src/vault.mojo").path),
            ("vault dir", vaultDir),
        ])
        // Per-ask transcript: the CLI names it (timestamp + question slug) and the
        // enclave harness appends the outside-model prompt + program to it.
        let session = newSessionLog(for: question)
        set("session transcript → \(session.path)")
        return try runLoggedScript(script.path, args, label: "ask",
                                   env: ["MILLFOLIO_SESSION_LOG": session.path])
    }

    /// Run the millfolio engine `index <path…>` over one or more files/folders.
    /// See runLoggedScript.
    public func runVaultIndex(paths: [String], force: Bool = false) throws -> Int32 {
        refreshServerRunning()
        ensureVaultShims()  // self-heal a wiped shared-toolchain lib/ before the vault binary dlopens it
        let script = try writeMillfolioScript()
        var args = ["index"] + paths
        if force { args.append("--force") }
        logRunDiagnostics(label: "index", launcher: script, args: args, probes: [
            ("millfolio launcher", script.path),
            ("millfolio dir (cwd)", millfolioDir.path),
            ("millfolio binary", millfolioBin.path),
            ("mojo compiler", millfolioMojoPrefix.appendingPathComponent("bin/mojo").path),
            ("paths", paths.joined(separator: " ")),
        ])
        return try runLoggedScript(script.path, args, label: "index")
    }

    /// Run the millfolio binary and CAPTURE its stdout — for the config get/set
    /// commands (`mill get/set amount-password`), which just read/write a small file
    /// in the data dir (no server, no streaming). Returns (trimmed stdout, exit code).
    public func runVaultConfig(_ args: [String]) -> (out: String, code: Int32) {
        ensureVaultShims()
        guard let script = try? writeMillfolioScript() else { return ("", 127) }
        return probe(script.path, args)
    }

    /// Dump everything useful for diagnosing a spawn failure: the exact command,
    /// whether each dependency path exists (and is executable), the launcher's
    /// contents (which set PATH/CONDA_PREFIX/MODULAR_HOME), and the inherited PATH.
    private func logRunDiagnostics(label: String, launcher: URL, args: [String], probes: [(String, String)]) {
        let fm = FileManager.default
        vlog("\n===== millfolio \(label) — \(Self.stamp()) =====")
        vlog("command: /bin/bash \(launcher.path) \(args.joined(separator: " "))")
        vlog("server running: \(serverRunning)")
        vlog("paths:")
        for (name, path) in probes {
            let tag = !fm.fileExists(atPath: path) ? "MISSING"
                    : fm.isExecutableFile(atPath: path) ? "exec" : "ok"
            vlog("  [\(tag)] \(name): \(path)")
        }
        if let body = try? String(contentsOf: launcher, encoding: .utf8) {
            vlog("launcher \(launcher.lastPathComponent):")
            for line in body.split(separator: "\n", omittingEmptySubsequences: false) {
                vlog("  | \(line)")
            }
        }
        vlog("inherited PATH: \(ProcessInfo.processInfo.environment["PATH"] ?? "(unset)")")
        vlog("----- child output -----")
    }

    /// Run `/bin/bash <script> <args…>` as a child, teeing its combined stdout and
    /// stderr to BOTH this terminal and the millfolio log. Streams live (so long runs
    /// show progress) and returns the exit status without throwing on nonzero.
    @discardableResult
    public func runLoggedScript(_ scriptPath: String, _ args: [String], label: String,
                                env extra: [String: String] = [:]) throws -> Int32 {
        let logFH = try? FileHandle(forWritingTo: ensureMillfolioLog())
        logFH?.seekToEndOfFile()
        let out = FileHandle.standardOutput
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [scriptPath] + args
        if !extra.isEmpty {
            var e = ProcessInfo.processInfo.environment
            for (k, v) in extra { e[k] = v }
            p.environment = e
        }
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        // standardInput is left inherited, so an interactive child still works.
        pipe.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            guard !d.isEmpty else { return }
            out.write(d)
            logFH?.write(d)
        }
        do {
            try p.run()
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            vlog("[\(label)] failed to launch /bin/bash: \(error)")
            try? logFH?.close()
            throw error
        }
        p.waitUntilExit()
        pipe.fileHandleForReading.readabilityHandler = nil
        let rest = pipe.fileHandleForReading.readDataToEndOfFile()
        if !rest.isEmpty { out.write(rest); logFH?.write(rest) }
        let code = p.terminationStatus
        vlog("[\(label)] exit status: \(code)")
        try? logFH?.close()
        return code
    }

    // ── self-update (CLI + components) ──────────────────────────────────────────
    /// Update the `millfolio` CLI via Homebrew (best-effort), then refresh the
    /// downloadable components — the inference-server engine, enclave, and the
    /// millfolio engine — to their latest releases. The pinned Mojo toolchains and the
    /// (multi-GB) model weights are preserved; only the source bundles are re-fetched
    /// and rebuilt. Progress streams through `onProgress`.
    public func selfUpdate(updateCLI: Bool = true) async throws {
        vlog("\n===== mill update — \(Self.stamp()) =====")
        if updateCLI {
            // Upgrade the CLI, then — if the binary actually changed — re-exec the
            // NEW binary to finish the component refresh. Otherwise the OLD installer
            // logic would run against the NEW bundle (their formats move together in
            // a release), which can fail on a crossing update (e.g. a file the old
            // installer expects that the new bundle dropped). Re-exec'ing pairs the
            // new installer with the new bundle.
            let before = brewCliVersion()
            updateHomebrewCLI()
            let after = brewCliVersion()
            if !after.isEmpty, after != before {
                set("Re-launching the updated CLI (\(after)) to finish…")
                reexecToFinishUpdate()  // execv; returns only if it couldn't re-exec
            }
        }

        // First reference to each component introduces it with a gloss; the
        // granular install steps below then use the short product name.
        // All components ship in ONE millfolio.zip now. Refresh by dropping the whole
        // unpacked bundle (incl. its .unpacked stamp) ONCE — `ensureBundle` (called by
        // installServer below) then re-fetches the latest bundle, and each install
        // rebuilds its component from it. Removing the per-component roots individually
        // would leave the stamp in place, so ensureBundle would skip the re-unpack and
        // the build would fail on a missing `inference-server`. Model weights (hf/) and
        // the Mojo toolchain (mojo/) live outside bundleRoot and are kept.
        try? FileManager.default.removeItem(at: bundleRoot)

        set("Refreshing engine, the inference server…")
        try await installServer()

        set("Refreshing enclave, the privacy agent harness…")
        try await installEnclaveEngine()

        set("Refreshing millfolio, the vault engine…")
        try await installMillfolioEngine()
        linkVaultShims()   // millfolio FFI shims → the shared toolchain lib (vault-run dlopen)
        primeVaultCompile()  // re-warm the vault compile cache (the nightly may have bumped)

        // The streaming app server (built on-device against enclave). A real build
        // error must fail the update, not be swallowed (else `mill update` falsely
        // reports success).
        set("Refreshing millfolio app server…")
        try await installAppServer()

        vlog("update complete")
    }

    /// Upgrade the CLI via Homebrew if it's installed that way. Best-effort: if brew
    /// or the formula isn't present, log it and carry on (components still refresh).
    private func updateHomebrewCLI() {
        let brew = ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
        guard let brew else {
            set("• Homebrew not found — skipping CLI self-update")
            vlog("brew not found at /opt/homebrew or /usr/local; skipped CLI self-update")
            return
        }
        set("Updating the millfolio CLI via Homebrew…")
        _ = try? run(brew, ["update"])   // refresh tap metadata (non-fatal if offline)
        do {
            let out = try run(brew, ["upgrade", "millfolio/tap/mill"])
            vlog("brew upgrade:\n\(out)")
            set("✓ CLI updated (takes effect next run)")
        } catch {
            // `brew upgrade` reports nonzero when nothing to do or the formula isn't
            // installed via brew — neither is fatal to a component refresh.
            vlog("brew upgrade (non-fatal): \(humanError(error))")
            set("• CLI not upgraded via Homebrew (already latest, or not a brew install)")
        }
    }

    /// Replace this process with the freshly-upgraded `mill` binary running
    /// `update --skip-cli`, so the component refresh runs with the NEW installer
    /// logic (paired with the NEW bundle). On success this never returns. Best-effort:
    /// if the brew-managed binary can't be found or `execv` fails, it returns and the
    /// caller finishes the refresh inline with the current binary.
    private func reexecToFinishUpdate() {
        let mill = ["/opt/homebrew/bin/mill", "/usr/local/bin/mill"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
        guard let mill else {
            vlog("re-exec: brew-managed mill not found; finishing with the current binary")
            return
        }
        vlog("re-exec: \(mill) update --skip-cli")
        let args: [String] = [mill, "update", "--skip-cli"]
        var cargs: [UnsafeMutablePointer<CChar>?] = args.map { strdup($0) }
        cargs.append(nil)
        execv(mill, &cargs)
        // Only reached if execv failed — fall back to an inline refresh.
        vlog("re-exec failed (execv): \(String(cString: strerror(errno))); finishing inline")
        for p in cargs where p != nil { free(p) }
    }

    // ── component versions ──────────────────────────────────────────────────────
    // millfolio ships as ONE bundle (millfolio.zip) cut from a single release tag,
    // alongside the `mill` CLI built from the same tag. So there are no independent
    // per-component versions: an installed component is, by construction, at the
    // CLI's version. `mill version` shows that version per installed component (and
    // "—" for anything not installed yet). The CLI version comes from Homebrew.

    /// The CLI's own version, from Homebrew ("" if not a brew install).
    private func brewCliVersion() -> String {
        guard let brew = ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]
            .first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else { return "" }
        // Two channels: prod is `millfolio/tap/mill` (binary `mill`), dev is
        // `millfolio/tap/mill-dev` (binary `mill-dev`). Prefer the formula matching how
        // THIS binary was invoked, so a dev CLI reads the dev version (and thus fetches
        // the dev bundle). If only one is installed, that one wins.
        let exe = (CommandLine.arguments.first.map { ($0 as NSString).lastPathComponent }) ?? "mill"
        let formulae = exe == "mill-dev"
            ? ["millfolio/tap/mill-dev", "millfolio/tap/mill"]
            : ["millfolio/tap/mill", "millfolio/tap/mill-dev"]
        for formula in formulae {
            guard let out = try? run(brew, ["list", "--versions", formula]) else { continue }
            let toks = out.split(whereSeparator: { $0 == " " || $0 == "\n" }).map(String.init)
            if toks.count >= 2 { return "v" + toks.last! }
        }
        return ""
    }

    /// Installed versions of millfolio + its components (label, version) for display.
    /// Every installed component is at the release/bundle version (the CLI's), so we
    /// show that; components not yet installed show "—". A non-Homebrew (dev) install
    /// has no tag, so installed components read "installed" instead of a version.
    public func componentVersions() -> [(String, String)] {
        let cli = brewCliVersion()
        let ver = cli.isEmpty ? "installed" : cli
        func v(_ installed: Bool) -> String { installed ? ver : "—" }
        return [
            ("cli (millfolio)", cli.isEmpty ? "—" : cli),
            ("inference server", v(isServerInstalled)),
            ("enclave", v(isEnclaveInstalled)),
            ("vault engine", v(isMillfolioInstalled)),
            ("app web server", v(isAppServerInstalled)),
        ]
    }

    // ── phase / progress sink ───────────────────────────────────────────────────
    private func set(_ msg: String) {
        phase = .running(msg)
        onProgress?(msg)
    }
    private func set(done: Bool) { phase = .done }
    private func set(failed msg: String) { phase = .failed(msg) }
}

public enum BootstrapError: Error, CustomStringConvertible {
    case step(String, String)
    case preflight(String)
    public var description: String {
        switch self {
        case .step(let s, let m): return "\(s): \(m)"
        case .preflight(let m): return m
        }
    }
}

func humanError(_ error: Error) -> String {
    if let b = error as? BootstrapError { return b.description }
    return (error as NSError).localizedDescription
}

/// Captures a single HTTP redirect's `Location` instead of following it — used to read
/// the tag out of GitHub's `…/releases/latest` → `…/releases/tag/vX.Y.Z` redirect
/// without an API call. Not `@MainActor`: URLSession invokes it on its own delegate
/// queue.
private final class RedirectCatcher: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)   // stop here; we only want the Location header
    }
}
