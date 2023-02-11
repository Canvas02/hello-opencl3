<!--
 Copyright 2023 Canvas02 <Canvas02@protonmail.com>.
 SPDX-License-Identifier: MIT
-->

# hello-opencl3

Testing opencl3 on rust

# Building

This example expects to find the `OPENCL_SDK` environment variable which points to an installation of the [KhronosGroup OpenCL-SDK](https://github.com/KhronosGroup/OpenCL-SDK),
if it fails it tries to find `OCL_ROOT` (from an [GPUOpen OCL-SDL](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/) installation) else the build fails
