// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "RemindersHelper",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "reminders-helper",
            path: "Sources/RemindersHelper",
            exclude: ["Info.plist"],
            linkerSettings: [
                // Embed the usage description so macOS can show the Reminders prompt.
                .unsafeFlags([
                    "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/RemindersHelper/Info.plist",
                ])
            ]
        )
    ]
)
