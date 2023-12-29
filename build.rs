//! Build specs for different LA systems

/// Links to Accelerate framework on MacOS running
#[cfg(target_os = "macos")]
fn main() {
    #[cfg(feature = "intel-mkl-system")]
    println!("cargo:rustc-link-lib=framework=Accelerate");
}

fn main() {}
