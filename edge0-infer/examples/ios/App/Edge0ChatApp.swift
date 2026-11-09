import SwiftUI
@main struct Edge0ChatApp: App {
    @StateObject private var model = ChatViewModel()
    @Environment(\.scenePhase) private var phase
    var body: some Scene {
        WindowGroup { ContentView(model: model).onChange(of: phase) { _, value in if value == .background { model.background() }; if value == .active { model.foreground() } } }
    }
}
