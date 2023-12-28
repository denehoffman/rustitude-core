#[cfg(any(feature = "openblas-system", feature = "netlib-system"))]
fn main() {}

#[cfg(not(any(feature = "openblas-system", feature = "netlib-system")))]
fn main() {
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
