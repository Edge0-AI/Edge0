import Foundation

public enum ExampleFixtureCleanup {
    public static let allowedNames = ["tiny-a", "tiny-b", "incomplete", "incomplete-example-check"]
    public static func run(in assets: URL, files: FileManager = .default) throws -> [String] {
        let attributes = try files.attributesOfItem(atPath: assets.path)
        guard attributes[.type] as? FileAttributeType == .typeDirectory else {
            throw NSError(domain: "Edge0ExampleCleanup", code: 1, userInfo: [NSLocalizedDescriptionKey: "Assets root must be a real directory, not a symbolic link"])
        }
        func removeWithoutFollowingLinks(_ item: URL) throws {
            let attributes = try files.attributesOfItem(atPath: item.path)
            if attributes[.type] as? FileAttributeType == .typeDirectory {
                for child in try files.contentsOfDirectory(at: item, includingPropertiesForKeys: nil) { try removeWithoutFollowingLinks(child) }
            }
            // Every link is unlinked as an item; its target is never enumerated.
            try files.removeItem(at: item)
        }
        for name in allowedNames {
            let item = assets.appendingPathComponent(name)
            do { _ = try files.attributesOfItem(atPath: item.path) }
            catch CocoaError.fileReadNoSuchFile { continue }
            try removeWithoutFollowingLinks(item)
        }
        return try files.contentsOfDirectory(atPath: assets.path).sorted()
    }
    public static func runStatusCheck(in assets: URL, badName: String?, includeTiny: Bool, files: FileManager = .default) throws {
        let attributes = try files.attributesOfItem(atPath: assets.path)
        guard attributes[.type] as? FileAttributeType == .typeDirectory else { throw CocoaError(.fileReadInvalidFileName) }
        var names: [String] = includeTiny ? ["tiny-a"] : []
        if let badName {
            let prefix = "app-status-bad-assets-"
            guard badName.hasPrefix(prefix), UUID(uuidString: String(badName.dropFirst(prefix.count))) != nil else { throw CocoaError(.fileReadInvalidFileName) }
            names.append(badName)
        }
        func remove(_ item: URL) throws {
            let attributes = try files.attributesOfItem(atPath: item.path)
            if attributes[.type] as? FileAttributeType == .typeDirectory {
                for child in try files.contentsOfDirectory(at: item, includingPropertiesForKeys: nil) { try remove(child) }
            }
            try files.removeItem(at: item)
        }
        for name in names {
            let item = assets.appendingPathComponent(name)
            do { _ = try files.attributesOfItem(atPath: item.path) } catch CocoaError.fileReadNoSuchFile { continue }
            try remove(item)
        }
    }

}
