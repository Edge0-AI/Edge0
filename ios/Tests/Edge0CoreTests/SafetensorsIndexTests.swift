import Edge0Core
import Foundation
import Testing

@Test func parsesSafetensorAndSlicesAxisZero() throws {
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
    defer { try? FileManager.default.removeItem(at: tmp) }

    let headerObject: [String: Any] = [
        "x": [
            "dtype": "U32",
            "shape": [2, 2],
            "data_offsets": [0, 16],
        ]
    ]
    var header = try JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
    // Safetensors permits JSON whitespace padding.
    while header.count % 8 != 0 { header.append(0x20) }
    var headerLength = UInt64(header.count).littleEndian
    var file = withUnsafeBytes(of: &headerLength) { Data($0) }
    file.append(header)
    file.append(Data(repeating: 0xAB, count: 16))
    try file.write(to: tmp)

    let index = try SafetensorsIndex(fileURL: tmp)
    let tensor = try index.tensor(named: "x")
    #expect(tensor.shape == [2, 2])
    #expect(tensor.byteCount == 16)
    let second = try tensor.axisZeroSlice(1)
    #expect(second.shape == [2])
    #expect(second.byteCount == 8)
}
