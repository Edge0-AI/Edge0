plugins {
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    // compose compiler plugin version must equal Kotlin (2.0.21); Room via KSP; metrics via kotlinx-serialization
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.ksp) apply false
    alias(libs.plugins.kotlinx.serialization) apply false
}
