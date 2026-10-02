// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "Edge0Phone",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "Edge0Core", targets: ["Edge0Core"]),
        .library(name: "Edge0MLX", targets: ["Edge0MLX"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.31.4")
        ),
    ],
    targets: [
        .target(name: "Edge0Core"),
        .target(
            name: "Edge0MLX",
            dependencies: [
                "Edge0Core",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "Edge0MLXTests", dependencies: ["Edge0MLX", "Edge0Core"], resources: [.copy("Fixtures")]),
        .testTarget(name: "Edge0CoreTests", dependencies: ["Edge0Core"]),
    ]
)
