use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../..").canonicalize().unwrap_or(manifest_dir.clone());
    let preferred = repo_root.join("mamba/custom_op/third_party/onnxruntime-linux-x64-1.16.3/lib");
    let fallback = repo_root.join("custom_op/third_party/onnxruntime-linux-x64-1.16.3/lib");
    let lib_dir = if preferred.is_dir() {
        preferred
    } else if fallback.is_dir() {
        fallback
    } else {
        panic!(
            "onnxruntime lib dir not found; expected {} or {}",
            preferred.display(),
            fallback.display()
        );
    };
    let lib_dir = lib_dir.canonicalize().unwrap_or(lib_dir);

    // onnxruntime-sys build script is disabled; link is fully managed here.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    let rpath = lib_dir.display().to_string();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", rpath);
}
