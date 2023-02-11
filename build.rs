// Copyright 2023 Canvas02 <Canvas02@protonmail.com>.
// SPDX-License-Identifier: MIT

use cfg_if::cfg_if;

fn main() {
    // println!(r"cargo:rustc-link-search=C:\Libs\_SDKs\OpenCL-SDK\lib");

    if let Some(path) = option_env!("OPENCL_SDK") {
        eprintln!("Using KhronosGroup OpenCL-SDK");

        dbg!(&path);
        println!(r"cargo:rustc-link-search={}/lib", path);
    } else if let Some(path) = option_env!("OCL_ROOT") {
        eprintln!("Using AMD OCL_SDK_Light");

        dbg!(&path);

        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                let arch = "x86_64";
            } else if #[cfg(target_arch = "x86")] {
                let arch = "x86";
            } else {
                panic!("OCL_SDK_Light only supports x86 and x86_64");
            }
        }

        println!(r"cargo:rustc-link-search={}/lib/{}", path, arch);
    } else {
        panic!("No OpenCL ICD found");
    }

    // unimplemented!()
}
