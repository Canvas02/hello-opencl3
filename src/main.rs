// Copyright 2023 Canvas02 <Canvas02@protonmail.com>.
// SPDX-License-Identifier: MIT

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
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY},
    program::Program,
    types::{cl_event, cl_float, CL_NON_BLOCKING},
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

    let z = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, ARRAY_SIZE, ptr::null_mut())
            .map_err(|err| format!("Failed to create buffer: {}", err.to_string()))
            .unwrap()
    };

    let x_write_event = unsafe {
        queue
            .enqueue_write_buffer(&mut x, CL_NON_BLOCKING, 0, &ones, &[])
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

