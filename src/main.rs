// Copyright 2023 Canvas02 <Canvas02@protonmail.com>.
// SPDX-License-Identifier: MIT

#![allow(unused)]

const PROGRAM_SOURCE: &str = r#"
kernel void saxpy_float (global float* z,
    global float const* x,
    global float const* y,
    float a)
{
    const size_t i = get_global_id(0);
    z[i] = a*x[i] + y[i];
}"#;

const KERNEL_NAME: &str = "saxpy_float";

use std::ptr;

use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_SIZE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY},
    program::Program,
    types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING},
};

// From https://github.com/kenba/opencl3/blob/4619128df954ac3aa1f2af7774c543f3be808b6c/examples/basic.rs
fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();

    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
        .expect("get_all_devices failed")
        .first()
        .expect("No device found");
    tracing::debug!("Found device: {:p}", device_id);

    let device = Device::new(device_id);
    tracing::debug!("Constructed device");

    let context = Context::from_device(&device)
        .map_err(|err| format!("Context::from_device failed: {}", err.to_string()))
        .unwrap();
    tracing::debug!("Constructed context: {:#?}", device);

    let queue =
        CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
            .map_err(|err| format!("Failed to create queue: {}", err.to_string()))
            .unwrap();
    let queue_size = {
        if let Ok(size) = queue
            .size()
            .map_err(|err| format!("Failed to get queue size: {}", err.to_string()))
        {
            Some(size)
        } else {
            None
        }
    };

    tracing::debug!("Created queue with size ({:?})", queue_size);

    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .map_err(|err| {
            format!(
                "Program::create_and_build_from_source failed: {}",
                err.to_string()
            )
        })
        .unwrap();

    let kernel = Kernel::create(&program, KERNEL_NAME)
        .map_err(|err| format!("Failed to create kernel: {}", err.to_string()))
        .unwrap();

    tracing::debug!(
        "Created program + kernel ({}) with source:\n{}",
        KERNEL_NAME,
        PROGRAM_SOURCE
    );

    const ARRAY_SIZE: usize = 1024;
    let ones: [cl_float; ARRAY_SIZE] = [1.0; ARRAY_SIZE];
    let sums: [cl_float; ARRAY_SIZE] = {
        let mut sums: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
        for i in 0..ARRAY_SIZE {
            sums[i] = 1.0 + 1.0 * i as cl_float;
        }

        sums
    };

    let mut x = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())
            .map_err(|err| format!("Failed to create buffer: {}", err.to_string()))
            .unwrap()
    };

    let mut y = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, ARRAY_SIZE, ptr::null_mut())
            .map_err(|err| format!("Failed to create buffer: {}", err.to_string()))
            .unwrap()
    };

    let mut z = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())
            .map_err(|err| format!("Failed to create buffer: {}", err.to_string()))
            .unwrap()
    };

    let x_write_event = unsafe {
        queue
            .enqueue_write_buffer(&mut x, CL_NON_BLOCKING, 0, &sums, &[])
            .map_err(|err| format!("Failed to write to buffer: {}", err.to_string()))
            .unwrap()
    };

    let y_write_event = unsafe {
        queue
            .enqueue_write_buffer(&mut y, CL_NON_BLOCKING, 0, &sums, &[])
            .map_err(|err| format!("Failed to write to buffer: {}", err.to_string()))
            .unwrap()
    };

    let a: cl_float = 300.0;

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&z)
            .set_arg(&x)
            .set_arg(&y)
            .set_arg(&a)
            .set_global_work_size(ARRAY_SIZE)
            .set_wait_event(&x_write_event)
            .set_wait_event(&y_write_event)
            .enqueue_nd_range(&queue)
    }
    .map_err(|err| format!("Failed to execute kernel: {}", err.to_string()))
    .unwrap();

    let mut events = Vec::<cl_event>::default();
    events.push(kernel_event.get());

    let mut result: [cl_float; ARRAY_SIZE] = [0.0; ARRAY_SIZE];
    let read_event =
        unsafe { queue.enqueue_read_buffer(&z, CL_NON_BLOCKING, 0, &mut result, &events) }
            .map_err(|err| format!("Failed to read buffer: {}", err.to_string()))
            .unwrap();

    read_event
        .wait()
        .map_err(|err| format!("Failed to wait to read buffer: {}", err.to_string()))
        .unwrap();

    println!("results front: {}", result[0]);
    println!("results back: {}", result[ARRAY_SIZE - 1]);

    let start_time = kernel_event
        .profiling_command_start()
        .map_err(|err| format!("Failed to start profiling command: {}", err.to_string()))
        .unwrap();

    let end_time = kernel_event
        .profiling_command_end()
        .map_err(|err| format!("Failed to end profiling command: {}", err.to_string()))
        .unwrap();

    let duration = end_time - start_time;
    tracing::info!("Kernel execution time (ns): {}", duration);

    Ok(())
}

// From https://stackoverflow.com/a/24336429/16854783
/*
fn get_cl_error_name(error_code: i32) -> &'static str {
    match error_code {
        // run-time and JIT compiler errors
        0 => "CL_SUCCESS",
        -1 => "CL_DEVICE_NOT_FOUND",
        -2 => "CL_DEVICE_NOT_AVAILABLE",
        -3 => "CL_COMPILER_NOT_AVAILABLE",
        -4 => "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        -5 => "CL_OUT_OF_RESOURCES",
        -6 => "CL_OUT_OF_HOST_MEMORY",
        -7 => "CL_PROFILING_INFO_NOT_AVAILABLE",
        -8 => "CL_MEM_COPY_OVERLAP",
        -9 => "CL_IMAGE_FORMAT_MISMATCH",
        -10 => "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        -11 => "CL_BUILD_PROGRAM_FAILURE",
        -12 => "CL_MAP_FAILURE",
        -13 => "CL_MISALIGNED_SUB_BUFFER_OFFSET",
        -14 => "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
        -15 => "CL_COMPILE_PROGRAM_FAILURE",
        -16 => "CL_LINKER_NOT_AVAILABLE",
        -17 => "CL_LINK_PROGRAM_FAILURE",
        -18 => "CL_DEVICE_PARTITION_FAILED",
        -19 => "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",

        // compile-time errors
        -30 => "CL_INVALID_VALUE",
        -31 => "CL_INVALID_DEVICE_TYPE",
        -32 => "CL_INVALID_PLATFORM",
        -33 => "CL_INVALID_DEVICE",
        -34 => "CL_INVALID_CONTEXT",
        -35 => "CL_INVALID_QUEUE_PROPERTIES",
        -36 => "CL_INVALID_COMMAND_QUEUE",
        -37 => "CL_INVALID_HOST_PTR",
        -38 => "CL_INVALID_MEM_OBJECT",
        -39 => "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        -40 => "CL_INVALID_IMAGE_SIZE",
        -41 => "CL_INVALID_SAMPLER",
        -42 => "CL_INVALID_BINARY",
        -43 => "CL_INVALID_BUILD_OPTIONS",
        -44 => "CL_INVALID_PROGRAM",
        -45 => "CL_INVALID_PROGRAM_EXECUTABLE",
        -46 => "CL_INVALID_KERNEL_NAME",
        -47 => "CL_INVALID_KERNEL_DEFINITION",
        -48 => "CL_INVALID_KERNEL",
        -49 => "CL_INVALID_ARG_INDEX",
        -50 => "CL_INVALID_ARG_VALUE",
        -51 => "CL_INVALID_ARG_SIZE",
        -52 => "CL_INVALID_KERNEL_ARGS",
        -53 => "CL_INVALID_WORK_DIMENSION",
        -54 => "CL_INVALID_WORK_GROUP_SIZE",
        -55 => "CL_INVALID_WORK_ITEM_SIZE",
        -56 => "CL_INVALID_GLOBAL_OFFSET",
        -57 => "CL_INVALID_EVENT_WAIT_LIST",
        -58 => "CL_INVALID_EVENT",
        -59 => "CL_INVALID_OPERATION",
        -60 => "CL_INVALID_GL_OBJECT",
        -61 => "CL_INVALID_BUFFER_SIZE",
        -62 => "CL_INVALID_MIP_LEVEL",
        -63 => "CL_INVALID_GLOBAL_WORK_SIZE",
        -64 => "CL_INVALID_PROPERTY",
        -65 => "CL_INVALID_IMAGE_DESCRIPTOR",
        -66 => "CL_INVALID_COMPILER_OPTIONS",
        -67 => "CL_INVALID_LINKER_OPTIONS",
        -68 => "CL_INVALID_DEVICE_PARTITION_COUNT",

        // extension errors
        -1000 => "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR",
        -1001 => "CL_PLATFORM_NOT_FOUND_KHR",
        -1002 => "CL_INVALID_D3D10_DEVICE_KHR",
        -1003 => "CL_INVALID_D3D10_RESOURCE_KHR",
        -1004 => "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR",
        -1005 => "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR",
        _ => "Unknown OpenCL error",
    }
}
*/
