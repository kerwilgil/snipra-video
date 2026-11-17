// Lanzador nativo de Snipra.app (macOS arm64).
//
// Responsabilidades:
//   1. Comprobar que el paquete está completo (Python, FFmpeg/FFprobe, backend, UI compilada).
//   2. Iniciar SOLO sus propios procesos: API (uvicorn) y worker, en su propio grupo de procesos.
//   3. Abrir la interfaz en el navegador predeterminado y mostrar errores comprensibles.
//   4. Detener esos procesos al salir. Nunca mata procesos ajenos ni usa puertos ocupados por otros.
//
// Datos persistentes:  ~/Library/Application Support/Snipra/   (fuera del bundle, sobrevive a actualizaciones)
// Registros:           ~/Library/Logs/Snipra/
// Caché:               ~/Library/Caches/Snipra/

import AppKit
import Darwin
import Foundation

// La extensión de navegador busca el backend en 8000 y después en 8001 (extension/background.js).
let preferredPorts: [Int] = [8000, 8001]
let healthTimeout: TimeInterval = 90

// MARK: - Logging

final class Log {
    private let url: URL
    private let lock = NSLock()

    init(url: URL) {
        self.url = url
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
    }

    func line(_ message: String) {
        lock.lock()
        defer { lock.unlock() }
        let stamp = ISO8601DateFormatter().string(from: Date())
        guard let data = "\(stamp) \(message)\n".data(using: .utf8),
              let handle = try? FileHandle(forWritingTo: url) else { return }
        handle.seekToEndOfFile()
        handle.write(data)
        try? handle.close()
    }
}

// MARK: - Paths

struct Paths {
    let bundle: URL
    let resources: URL
    let helpers: URL
    let python: URL
    let backendDir: URL
    let frontendDist: URL
    let fontsConf: URL
    let seedModelsHub: URL
    let dataDir: URL
    let storageDir: URL
    let logsDir: URL
    let cacheDir: URL

    static func resolve() throws -> Paths {
        let fm = FileManager.default
        let bundle = Bundle.main.bundleURL
        let resources = bundle.appendingPathComponent("Contents/Resources", isDirectory: true)
        let home = fm.homeDirectoryForCurrentUser
        let support = try fm.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                 appropriateFor: nil, create: true)
            .appendingPathComponent("Snipra", isDirectory: true)
        return Paths(
            bundle: bundle,
            resources: resources,
            helpers: bundle.appendingPathComponent("Contents/Helpers", isDirectory: true),
            python: resources.appendingPathComponent("python/bin/python3.11"),
            backendDir: resources.appendingPathComponent("app/backend", isDirectory: true),
            frontendDist: resources.appendingPathComponent("app/frontend/dist", isDirectory: true),
            fontsConf: resources.appendingPathComponent("fonts/fonts.conf"),
            seedModelsHub: resources.appendingPathComponent("models/hub", isDirectory: true),
            dataDir: support,
            storageDir: support.appendingPathComponent("storage", isDirectory: true),
            logsDir: home.appendingPathComponent("Library/Logs/Snipra", isDirectory: true),
            cacheDir: home.appendingPathComponent("Library/Caches/Snipra", isDirectory: true)
        )
    }

    var ffmpeg: URL { helpers.appendingPathComponent("ffmpeg") }
    var ffprobe: URL { helpers.appendingPathComponent("ffprobe") }
    var modelsHubCache: URL { dataDir.appendingPathComponent("models/hub", isDirectory: true) }

    func requiredMissing() -> [String] {
        let fm = FileManager.default
        let required: [(String, String)] = [
            ("Python integrado", python.path),
            ("FFmpeg", ffmpeg.path),
            ("FFprobe", ffprobe.path),
            ("Backend", backendDir.appendingPathComponent("app/main.py").path),
            ("Interfaz compilada", frontendDist.appendingPathComponent("index.html").path),
            ("Configuración de fuentes", fontsConf.path),
        ]
        return required.filter { !fm.fileExists(atPath: $0.1) }.map { $0.0 }
    }
}

// MARK: - Procesos propios

final class Service {
    let name: String
    let logURL: URL
    private(set) var pid: pid_t = 0
    private(set) var exitStatus: Int32?

    init(name: String, logURL: URL) {
        self.name = name
        self.logURL = logURL
    }

    var isRunning: Bool { pid > 0 && exitStatus == nil }

    /// Lanza el proceso en un grupo propio para poder detenerlo junto con sus hijos.
    func start(executable: String, arguments: [String], environment: [String: String]) throws {
        var fileActions: posix_spawn_file_actions_t? = nil
        posix_spawn_file_actions_init(&fileActions)
        defer { posix_spawn_file_actions_destroy(&fileActions) }
        posix_spawn_file_actions_addopen(&fileActions, 0, "/dev/null", O_RDONLY, 0)
        let logPath = logURL.path
        posix_spawn_file_actions_addopen(&fileActions, 1, logPath, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        posix_spawn_file_actions_adddup2(&fileActions, 1, 2)

        var attributes: posix_spawnattr_t? = nil
        posix_spawnattr_init(&attributes)
        defer { posix_spawnattr_destroy(&attributes) }
        // El launcher ignora SIGTERM/SIGINT/SIGHUP para cerrar en orden; esas
        // disposiciones se heredan por exec. Los hijos vuelven a la acción por
        // defecto, o no terminarían con la señal de parada.
        var defaultSignals = sigset_t()
        sigemptyset(&defaultSignals)
        sigaddset(&defaultSignals, SIGTERM)
        sigaddset(&defaultSignals, SIGINT)
        sigaddset(&defaultSignals, SIGHUP)
        posix_spawnattr_setsigdefault(&attributes, &defaultSignals)
        // libdispatch bloquea estas señales en el hilo del launcher; la máscara también se hereda.
        var unblocked = sigset_t()
        sigemptyset(&unblocked)
        posix_spawnattr_setsigmask(&attributes, &unblocked)
        posix_spawnattr_setflags(&attributes, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK))
        posix_spawnattr_setpgroup(&attributes, 0)

        let argv = cStringArray([executable] + arguments)
        let envp = cStringArray(environment.map { "\($0.key)=\($0.value)" })
        defer {
            argv.forEach { free($0) }
            envp.forEach { free($0) }
        }

        var newPid: pid_t = 0
        let status = posix_spawn(&newPid, executable, &fileActions, &attributes, argv, envp)
        guard status == 0 else {
            throw LauncherError.spawnFailed(name, String(cString: strerror(status)))
        }
        pid = newPid
        exitStatus = nil
    }

    /// Recoge el estado si el proceso ya terminó (evita zombis y permite detectar caídas).
    func poll() {
        guard pid > 0, exitStatus == nil else { return }
        var status: Int32 = 0
        let result = waitpid(pid, &status, WNOHANG)
        if result == pid {
            exitStatus = (status & 0x7f) == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f)
        }
    }

    /// Envía SIGTERM al grupo propio y, si no sale a tiempo, SIGKILL. Solo afecta a este grupo.
    func stop(grace: TimeInterval = 8, log: Log? = nil) {
        guard pid > 0 else { return }
        poll()
        if exitStatus == nil {
            log?.line("\(name): SIGTERM al grupo \(pid)")
            kill(-pid, SIGTERM)
            let deadline = Date().addingTimeInterval(grace)
            while exitStatus == nil && Date() < deadline {
                usleep(100_000)
                poll()
            }
        }
        if exitStatus == nil {
            log?.line("\(name): sin salida tras \(grace) s; SIGKILL al grupo \(pid)")
            kill(-pid, SIGKILL)
            let killDeadline = Date().addingTimeInterval(3)
            while exitStatus == nil && Date() < killDeadline {
                usleep(50_000)
                poll()
            }
            if exitStatus == nil { exitStatus = 137 }
        }
    }
}

enum LauncherError: LocalizedError {
    case missingComponents([String])
    case spawnFailed(String, String)
    case noFreePort
    case alreadyRunning
    case startupFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingComponents(let items):
            return "El paquete de Snipra está incompleto (falta: \(items.joined(separator: ", "))). Vuelve a descargar e instalar Snipra."
        case .spawnFailed(let name, let reason):
            return "No se pudo iniciar \(name): \(reason)."
        case .noFreePort:
            return "Los puertos 8000 y 8001 están ocupados por otros programas. Cierra el programa que los usa y vuelve a abrir Snipra."
        case .alreadyRunning:
            return "Snipra ya está en ejecución."
        case .startupFailed(let reason):
            return reason
        }
    }
}

// MARK: - Utilidades

/// Copia de cadenas C terminada en NULL para posix_spawn. El llamador libera cada elemento.
func cStringArray(_ items: [String]) -> [UnsafeMutablePointer<CChar>?] {
    var result: [UnsafeMutablePointer<CChar>?] = []
    for item in items {
        result.append(strdup(item))
    }
    result.append(nil)
    return result
}

func portIsOpen(_ port: Int) -> Bool {
    let sock = socket(AF_INET, SOCK_STREAM, 0)
    guard sock >= 0 else { return false }
    defer { close(sock) }
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(port).bigEndian
    addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    return withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            connect(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
        }
    }
}

/// Puerto libre entre los preferidos. Nunca cierra ni reutiliza un servicio ajeno.
func choosePort() -> Int? {
    preferredPorts.first { !portIsOpen($0) }
}

/// GET síncrono con timeout (se llama desde un hilo de fondo).
func httpGet(_ url: URL, timeout: TimeInterval = 3) -> (status: Int, body: Data)? {
    var request = URLRequest(url: url)
    request.timeoutInterval = timeout
    var result: (Int, Data)?
    let done = DispatchSemaphore(value: 0)
    URLSession.shared.dataTask(with: request) { data, response, _ in
        if let http = response as? HTTPURLResponse, let data = data {
            result = (http.statusCode, data)
        }
        done.signal()
    }.resume()
    _ = done.wait(timeout: .now() + timeout + 2)
    return result.map { (status: $0.0, body: $0.1) }
}

func commandLine(of pid: pid_t) -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/ps")
    process.arguments = ["-p", String(pid), "-o", "command="]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    do {
        try process.run()
        process.waitUntilExit()
    } catch {
        return ""
    }
    return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
}

// MARK: - Estado de ejecución (runtime.json) y bloqueo de instancia única

struct RuntimeRecord: Codable {
    var launcherPid: Int32
    var port: Int
    var backendPid: Int32
    var workerPid: Int32
    var version: String
}

final class InstanceLock {
    private var descriptor: Int32 = -1

    /// Devuelve false si ya hay otra instancia del lanzador.
    func acquire(at url: URL) -> Bool {
        // O_CLOEXEC: los procesos hijos no deben heredar el lock.
        descriptor = open(url.path, O_RDWR | O_CREAT | O_CLOEXEC, 0o644)
        guard descriptor >= 0 else { return true }
        return flock(descriptor, LOCK_EX | LOCK_NB) == 0
    }

    func release() {
        if descriptor >= 0 { flock(descriptor, LOCK_UN); close(descriptor); descriptor = -1 }
    }
}

// MARK: - Controlador principal

final class AppController: NSObject, NSApplicationDelegate {
    private var paths: Paths!
    private var log: Log!
    private let instanceLock = InstanceLock()
    private var backend: Service?
    private var worker: Service?
    private var port: Int = 0
    private var monitorTimer: Timer?
    private var openedBrowserOnce = false
    private var isStopping = false
    private var signalSources: [DispatchSourceSignal] = []

    private var window: NSWindow!
    private var statusLabel: NSTextField!
    private var detailLabel: NSTextField!
    private var openButton: NSButton!
    private var logsButton: NSButton!
    private var stopButton: NSButton!

    private var runtimeURL: URL { paths.dataDir.appendingPathComponent("runtime.json") }
    private var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0.0"
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Snipra mantiene servicios vivos: no dejar que macOS cierre la app por inactividad.
        ProcessInfo.processInfo.disableAutomaticTermination("Servicios de Snipra en ejecución")
        ProcessInfo.processInfo.disableSuddenTermination()
        installSignalHandlers()
        buildWindow()
        do {
            paths = try Paths.resolve()
            try [paths.dataDir, paths.storageDir, paths.logsDir, paths.cacheDir]
                .forEach { try FileManager.default.createDirectory(at: $0, withIntermediateDirectories: true) }
            log = Log(url: paths.logsDir.appendingPathComponent("launcher.log"))
        } catch {
            fail("No se pudieron crear las carpetas de Snipra: \(error.localizedDescription)")
            return
        }
        log.line("== Snipra \(version) (lanzador) ==")

        guard instanceLock.acquire(at: paths.dataDir.appendingPathComponent("launcher.lock")) else {
            log.line("Otra instancia ya está activa; se reutiliza.")
            reuseRunningInstance()
            return
        }

        let missing = paths.requiredMissing()
        if !missing.isEmpty {
            fail(LauncherError.missingComponents(missing).localizedDescription)
            return
        }

        DispatchQueue.global(qos: .userInitiated).async { self.startServices() }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        logTerminationRequest("applicationShouldTerminate")
        stopServices()
        return .terminateNow
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        logTerminationRequest("ventana cerrada")
        return true
    }

    /// Registra quién pidió el cierre: evento Apple (clase/id), PID del remitente y evento de la app.
    /// Sin esto, un cierre no pedido por el usuario no se puede atribuir.
    private func logTerminationRequest(_ origin: String) {
        let event = NSAppleEventManager.shared().currentAppleEvent
        let fourCC = { (code: UInt32) -> String in
            String(bytes: [UInt8((code >> 24) & 0xff), UInt8((code >> 16) & 0xff),
                           UInt8((code >> 8) & 0xff), UInt8(code & 0xff)], encoding: .macOSRoman) ?? "????"
        }
        let eventText = event.map { "\(fourCC($0.eventClass))/\(fourCC($0.eventID))" } ?? "sin-evento-apple"
        let senderPid = event?.attributeDescriptor(forKeyword: 0x73706964)?.int32Value ?? -1  // 'spid'
        let appEvent = NSApp.currentEvent.map { "tipo \($0.type.rawValue)" } ?? "sin-evento-app"
        log?.line("Cierre solicitado (\(origin)): evento=\(eventText) emisor_pid=\(senderPid) \(appEvent)")
    }

    /// SIGTERM, SIGINT y SIGHUP piden un cierre ordenado: se detienen API y worker antes de salir.
    private func installSignalHandlers() {
        for signalNumber in [SIGTERM, SIGINT, SIGHUP] {
            signal(signalNumber, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: signalNumber, queue: .main)
            source.setEventHandler { [weak self] in
                self?.log?.line("Señal \(signalNumber) recibida: cierre ordenado.")
                NSApp.terminate(nil)
            }
            source.resume()
            signalSources.append(source)
        }
    }

    // MARK: Arranque

    private func startServices() {
        // 1) Limpiar restos de un lanzador anterior que murió sin detener sus procesos.
        cleanupOwnOrphans()

        // 2) Puerto: 8000 o 8001 libres. Si no, error claro (no se mata nada).
        guard let chosen = choosePort() else {
            return fail(LauncherError.noFreePort.localizedDescription)
        }
        port = chosen
        log.line("Puerto elegido: \(port)")
        setStatus("Preparando Snipra…", detail: "Comprobando componentes y modelos locales.")

        // 3) Modelo de transcripción incluido en el paquete (copia única a Application Support).
        seedWhisperModel()

        // 4) API.
        let env = environment()
        let api = Service(name: "API", logURL: paths.logsDir.appendingPathComponent("backend-api.log"))
        backend = api
        setStatus("Iniciando la API local…", detail: "Puerto \(port).")
        do {
            try api.start(executable: paths.python.path, arguments: [
                "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(port),
                "--no-access-log", "--timeout-graceful-shutdown", "5",
            ], environment: env)
        } catch {
            return fail(error.localizedDescription)
        }

        // Registro desde el primer instante: si el launcher muere, el siguiente arranque ve estos PIDs.
        writeRuntimeRecord()

        guard waitForHealth() else {
            stopServices()
            return fail("La API no respondió en 90 segundos. Revisa el registro: ~/Library/Logs/Snipra/backend-api.log")
        }

        // 5) Worker (toma trabajos de la misma base de datos).
        let worker = Service(name: "worker", logURL: paths.logsDir.appendingPathComponent("backend-worker.log"))
        self.worker = worker
        setStatus("Iniciando el procesador de trabajos…", detail: "Worker local.")
        do {
            try worker.start(executable: paths.python.path, arguments: ["-m", "app.worker.runner"], environment: env)
        } catch {
            stopServices()
            return fail(error.localizedDescription)
        }
        writeRuntimeRecord()
        Thread.sleep(forTimeInterval: 2.5)
        worker.poll()
        if !worker.isRunning {
            log.line("El worker terminó con estado \(worker.exitStatus ?? -1).")
            stopServices()
            return fail("El procesador de trabajos se detuvo al iniciar. Revisa ~/Library/Logs/Snipra/backend-worker.log")
        }

        writeRuntimeRecord()
        log.line("Servicios listos en http://127.0.0.1:\(port)/ (API pid \(api.pid), worker pid \(worker.pid))")
        DispatchQueue.main.async {
            self.setReady()
            self.startMonitoring()
        }
    }

    private func waitForHealth() -> Bool {
        let url = URL(string: "http://127.0.0.1:\(port)/api/health")!
        let deadline = Date().addingTimeInterval(healthTimeout)
        while Date() < deadline {
            backend?.poll()
            if backend?.isRunning != true {
                log.line("La API terminó durante el arranque (estado \(backend?.exitStatus ?? -1)).")
                return false
            }
            if let response = httpGet(url, timeout: 2), response.status == 200,
               let json = try? JSONSerialization.jsonObject(with: response.body) as? [String: Any],
               let binaries = json["binaries"] as? [String: Any],
               (binaries["ffmpeg"] as? [String: Any])?["available"] as? Bool == true,
               (binaries["ffprobe"] as? [String: Any])?["available"] as? Bool == true {
                return true
            }
            Thread.sleep(forTimeInterval: 0.5)
        }
        return false
    }

    private func seedWhisperModel() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: paths.seedModelsHub.path) else { return }
        do {
            try fm.createDirectory(at: paths.modelsHubCache, withIntermediateDirectories: true)
            let entries = try fm.contentsOfDirectory(atPath: paths.seedModelsHub.path)
            for entry in entries {
                let destination = paths.modelsHubCache.appendingPathComponent(entry)
                if !fm.fileExists(atPath: destination.path) {
                    try fm.copyItem(at: paths.seedModelsHub.appendingPathComponent(entry), to: destination)
                    log.line("Modelo de transcripción copiado: \(entry)")
                }
            }
        } catch {
            log.line("Aviso: no se pudo preparar el modelo incluido: \(error.localizedDescription)")
        }
    }

    /// Entorno mínimo y explícito: no hereda PATH, PYTHON* ni variables del desarrollador.
    private func environment() -> [String: String] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return [
            "HOME": home,
            "USER": NSUserName(),
            "LANG": "en_US.UTF-8",
            "LC_CTYPE": "UTF-8",
            "TMPDIR": NSTemporaryDirectory(),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONPATH": paths.backendDir.path,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "STORAGE_DIR": paths.storageDir.path,
            "FRONTEND_DIST_DIR": paths.frontendDist.path,
            "CORS_ORIGINS": "http://127.0.0.1:\(port),http://localhost:\(port)",
            "FFMPEG_PATH": paths.ffmpeg.path,
            "FFPROBE_PATH": paths.ffprobe.path,
            "FONTCONFIG_FILE": paths.fontsConf.path,
            "HF_HOME": paths.dataDir.appendingPathComponent("models").path,
            "HF_HUB_CACHE": paths.modelsHubCache.path,
            "XDG_CACHE_HOME": paths.cacheDir.path,
        ]
    }

    // MARK: Estado y registro

    private func writeRuntimeRecord() {
        let record = RuntimeRecord(launcherPid: getpid(), port: port,
                                   backendPid: backend?.pid ?? 0, workerPid: worker?.pid ?? 0,
                                   version: version)
        if let data = try? JSONEncoder().encode(record) {
            try? data.write(to: runtimeURL, options: .atomic)
        }
    }

    /// Detiene únicamente procesos registrados por un lanzador anterior y que siguen siendo de este paquete.
    private func cleanupOwnOrphans() {
        guard let data = try? Data(contentsOf: runtimeURL),
              let record = try? JSONDecoder().decode(RuntimeRecord.self, from: data) else { return }
        // Normalizar a NFC: la ruta con acentos puede llegar en NFD desde el sistema de archivos.
        let marker = paths.python.path.precomposedStringWithCanonicalMapping
        var orphans: [pid_t] = []
        for pid in [record.backendPid, record.workerPid] where pid > 0 {
            let command = commandLine(of: pid).precomposedStringWithCanonicalMapping
            guard kill(pid, 0) == 0, command.contains(marker) else { continue }
            log.line("Deteniendo proceso huérfano de Snipra (pid \(pid)).")
            kill(-pid, SIGTERM)
            orphans.append(pid)
        }
        // Espera al apagado gradual (uvicorn cierra con --timeout-graceful-shutdown 5) y libera el puerto.
        let deadline = Date().addingTimeInterval(7)
        while orphans.contains(where: { kill($0, 0) == 0 }) && Date() < deadline {
            usleep(200_000)
        }
        for pid in orphans where kill(pid, 0) == 0 {
            log.line("Huérfano pid \(pid) sin salida; SIGKILL a su grupo.")
            kill(-pid, SIGKILL)
        }
        try? FileManager.default.removeItem(at: runtimeURL)
    }

    private func startMonitoring() {
        monitorTimer?.invalidate()
        monitorTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { [weak self] _ in
            self?.checkServices()
        }
    }

    private func checkServices() {
        guard !isStopping else { return }
        backend?.poll()
        worker?.poll()
        if backend?.isRunning != true || worker?.isRunning != true {
            log.line("Un servicio se detuvo inesperadamente (API: \(backend?.exitStatus ?? -1), worker: \(worker?.exitStatus ?? -1)).")
            setStatus("Un servicio de Snipra se detuvo.",
                      detail: "Revisa los registros en ~/Library/Logs/Snipra y vuelve a abrir la aplicación.")
            monitorTimer?.invalidate()
        }
    }

    private func stopServices() {
        guard !isStopping else { return }
        isStopping = true
        monitorTimer?.invalidate()
        log.line("Deteniendo servicios propios…")
        worker?.stop(log: log)
        log.line("Worker detenido (estado \(worker?.exitStatus ?? -1)).")
        backend?.stop(log: log)
        log.line("API detenida (estado \(backend?.exitStatus ?? -1)).")
        try? FileManager.default.removeItem(at: runtimeURL)
        instanceLock.release()
        log.line("Servicios detenidos.")
    }

    private func reuseRunningInstance() {
        guard let data = try? Data(contentsOf: runtimeURL),
              let record = try? JSONDecoder().decode(RuntimeRecord.self, from: data),
              let url = URL(string: "http://127.0.0.1:\(record.port)/"),
              httpGet(URL(string: "http://127.0.0.1:\(record.port)/api/health")!, timeout: 2)?.status == 200 else {
            fail(LauncherError.alreadyRunning.localizedDescription + " Ciérrala antes de abrirla de nuevo.")
            return
        }
        NSWorkspace.shared.open(url)
        DispatchQueue.main.async { NSApp.terminate(nil) }
    }

    // MARK: Interfaz

    private func buildWindow() {
        let style: NSWindow.StyleMask = [.titled, .closable, .miniaturizable]
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 460, height: 230),
                          styleMask: style, backing: .buffered, defer: false)
        window.title = "Snipra \(version)"
        window.center()
        window.isReleasedWhenClosed = false

        let content = NSView(frame: window.contentRect(forFrameRect: window.frame))
        window.contentView = content

        statusLabel = NSTextField(labelWithString: "Iniciando Snipra…")
        statusLabel.font = .systemFont(ofSize: 15, weight: .semibold)
        statusLabel.frame = NSRect(x: 24, y: 160, width: 412, height: 24)

        detailLabel = NSTextField(wrappingLabelWithString: "")
        detailLabel.font = .systemFont(ofSize: 12)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.frame = NSRect(x: 24, y: 90, width: 412, height: 60)
        detailLabel.maximumNumberOfLines = 4

        openButton = NSButton(title: "Abrir Snipra", target: self, action: #selector(openUI))
        openButton.bezelStyle = .rounded
        openButton.keyEquivalent = "\r"
        openButton.isEnabled = false
        openButton.frame = NSRect(x: 24, y: 36, width: 130, height: 32)

        logsButton = NSButton(title: "Ver registros", target: self, action: #selector(openLogs))
        logsButton.bezelStyle = .rounded
        logsButton.frame = NSRect(x: 164, y: 36, width: 130, height: 32)

        stopButton = NSButton(title: "Detener Snipra", target: self, action: #selector(quit))
        stopButton.bezelStyle = .rounded
        stopButton.frame = NSRect(x: 306, y: 36, width: 130, height: 32)

        [statusLabel, detailLabel, openButton, logsButton, stopButton].forEach { content.addSubview($0) }
        window.makeKeyAndOrderFront(nil)
    }

    private func setStatus(_ status: String, detail: String) {
        DispatchQueue.main.async {
            self.statusLabel.stringValue = status
            self.detailLabel.stringValue = detail
        }
    }

    private func setReady() {
        statusLabel.stringValue = "Snipra está listo"
        detailLabel.stringValue = "Servicio local en http://127.0.0.1:\(port)/ . Tus datos se guardan en ~/Library/Application Support/Snipra."
        openButton.isEnabled = true
        if !openedBrowserOnce {
            openedBrowserOnce = true
            openUI()
        }
    }

    private func fail(_ message: String) {
        log?.line("ERROR: \(message)")
        DispatchQueue.main.async {
            self.statusLabel.stringValue = "Snipra no pudo iniciar"
            self.detailLabel.stringValue = message
            self.openButton.isEnabled = false
            let alert = NSAlert()
            alert.messageText = "Snipra no pudo iniciar"
            alert.informativeText = message
            alert.alertStyle = .critical
            alert.addButton(withTitle: "Ver registros")
            alert.addButton(withTitle: "Cerrar")
            if alert.runModal() == .alertFirstButtonReturn, let logs = self.paths?.logsDir {
                NSWorkspace.shared.open(logs)
            }
            NSApp.terminate(nil)
        }
    }

    @objc private func openUI() {
        guard port > 0, let url = URL(string: "http://127.0.0.1:\(port)/") else { return }
        NSWorkspace.shared.open(url)
    }

    @objc private func openLogs() {
        if let logs = paths?.logsDir { NSWorkspace.shared.open(logs) }
    }

    @objc private func quit() {
        NSApp.terminate(nil)
    }
}

// MARK: - Entrada

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let controller = AppController()
app.delegate = controller
app.run()
