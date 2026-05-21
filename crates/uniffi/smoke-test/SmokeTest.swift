// Phase 0 UniFFI smoke test. Validates that the Rust engine surface
// reaches Swift through UniFFI-generated bindings end to end.

import Foundation

@main
struct SmokeTest {
    static func main() {
        let version = engineVersion()
        print("sadda engine version from Swift: \(version)")
        guard !version.isEmpty else {
            print("ERROR: engine version is empty")
            exit(1)
        }
    }
}
