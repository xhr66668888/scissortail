fn main() {
    // The Windows x64 build starts on Windows 7 only when it is linked the way
    // scripts/build.sh does it: YY-Thunks and an ole32 redirect for
    // windows-core's combase imports, placed before every import library by
    // an lld-link wrapper (SCISSORTAIL_LINK_FIRST). Changing them relinks.
    println!("cargo:rerun-if-env-changed=SCISSORTAIL_LINK_FIRST");
    let var = |k: &str| std::env::var(k).unwrap_or_default();
    if var("CARGO_CFG_TARGET_OS") == "windows" && var("CARGO_CFG_TARGET_ARCH") == "x86_64" {
        let first = var("SCISSORTAIL_LINK_FIRST");
        if first.is_empty() {
            println!("cargo:warning=not built with scripts/build.sh: this program will not start on Windows 7/8");
        }
        for path in first.split(':').filter(|p| !p.is_empty()) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    tauri_build::build()
}
