#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define INDEX(X, Y) (Y + 1) * buffer_width + X + 1

int main(int argc, char **argv) {
  // Input parsing
  int iterations;
  cl_float diffusion_constant;
  if (argv[1][1] == 'n') {
    iterations = atoi((const char *)argv[1] + 2);
    diffusion_constant = atof((const char *)argv[2] + 2);
  } else {
    iterations = atoi((const char *)argv[2] + 2);
    diffusion_constant = atof((const char *)argv[1] + 2);
  }
  // Optional file specifier for testing

  char *file;
  if (argc == 4) {
    file = argv[3];
  } else {
    file = "./init";
  } // Open the files
  FILE *fp = fopen(file, "r");

  int height;
  int width;

  fscanf(fp, "%d %d", &width, &height);

  // TODO Handle arguments and file read

  // Set values should come from arguments and file in the future
  // const int height = 10000;
  // const int width = 10000;

  // Add padding to the buffers
  const int buffer_width = height + 2;
  const int buffer_height = width + 2;

  // Total size of a buffer
  const int sz = buffer_height * buffer_width;

  // Values used for reduction
  const int global_redsz = 1024;
  const int local_redsz = 32;
  const int nmb_redgps = global_redsz / local_redsz;

  // Allocate starting values for each of the buffers
  cl_float *a = calloc(buffer_width * buffer_height, sizeof(cl_float));
  cl_float *b = calloc(buffer_width * buffer_height, sizeof(cl_float));

  // Set values gathered from reading the file in buffer a
  int x, y;
  float value;
  for (int i = 0; fscanf(fp, "%d %d %f", &x, &y, &value) != EOF; i++) {
    // printf("Test %d: %f \n", i, value);
    a[INDEX(x, y)] = value;
  }

  // return 1;
  //  Place holder values to get some performance approxiamation
  //  for (size_t jx = 1; jx < buffer_height - 1; ++jx) {
  //    for (size_t ix = 1; ix < buffer_width - 1; ++ix) {
  //      a[jx * buffer_width + ix] = (float)ix * jx;
  //    }
  //  }

  // Useful when testing
  // a[buffer_width * 2 + 2] = 1000000.0;

  // BEGIN OPENCL BOILER PLATE

  cl_int error;

  cl_platform_id platform_id;
  cl_uint nmb_platforms;
  if (clGetPlatformIDs(2, &platform_id, &nmb_platforms) != CL_SUCCESS) {
    fprintf(stderr, "cannot get platform\n");
    return 1;
  }

  cl_device_id device_id;
  cl_uint nmb_devices;
  if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                     &nmb_devices) != CL_SUCCESS) {
    fprintf(stderr, "cannot get device\n");
    return 1;
  }

  cl_context context;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_id, 0};
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create context\n");
    return 1;
  }

  cl_command_queue command_queue;
  command_queue =
      clCreateCommandQueueWithProperties(context, device_id, NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create command queue\n");
    return 1;
  }
  // END OPENCL BOILER PLATE

  // BEGIN CREATE OPENCL PROGRAM BOILER PLATE
  // Read and create opencl program from source file
  char *opencl_program_src;
  {
    FILE *clfp = fopen("./diffusion.cl", "r");
    if (clfp == NULL) {
      fprintf(stderr, "could not load cl source code\n");
      return 1;
    }
    fseek(clfp, 0, SEEK_END);
    int clfsz = ftell(clfp);
    fseek(clfp, 0, SEEK_SET);
    opencl_program_src = (char *)malloc((clfsz + 1) * sizeof(char));
    fread(opencl_program_src, sizeof(char), clfsz, clfp);
    opencl_program_src[clfsz] = 0;
    fclose(clfp);
  }

  // Create the opencl program
  cl_program program;
  size_t src_len = strlen(opencl_program_src);
  program =
      clCreateProgramWithSource(context, 1, (const char **)&opencl_program_src,
                                (const size_t *)&src_len, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create program\n");
    return 1;
  }
  // free some memory used for storing source code
  free(opencl_program_src);

  error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  // Report if we failed to create program
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot build program. log:\n");

    size_t log_size = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);

    char *log = malloc(log_size * sizeof(char));
    if (log == NULL) {
      fprintf(stderr, "could not allocate memory\n");
      return 1;
    }

    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);

    fprintf(stderr, "%s\n", log);

    free(log);

    return 1;
  }

  // END CREATE OPENCL PROGRAM BOILER PLATE

  // Create the different kernels
  cl_kernel kernela = clCreateKernel(program, "heat_diffusion", &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create kernel\n");
    return 1;
  }
  cl_kernel kernelb = clCreateKernel(program, "heat_diffusion", &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create kernel\n");
    return 1;
  }
  cl_kernel kernel_reduction = clCreateKernel(program, "reduction", &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create kernel reduction\n");
    return 1;
  }
  cl_kernel kernel_heat_difference =
      clCreateKernel(program, "heat_difference", &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create kernel reduction\n");
    return 1;
  }

  // Create Read/Write buffers for the heat diffsuions
  cl_mem work_buffer_a, work_buffer_b, reduction_buffer;
  work_buffer_a = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      buffer_width * buffer_height * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create buffer a\n");
    return 1;
  }
  work_buffer_b = clCreateBuffer(
      context, CL_MEM_READ_WRITE,
      buffer_width * buffer_height * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create buffer b\n");
    return 1;
  }
  reduction_buffer = clCreateBuffer(
      context, CL_MEM_READ_WRITE, nmb_redgps * sizeof(cl_float), NULL, &error);
  if (error != CL_SUCCESS) {
    fprintf(stderr, "cannot create buffer c_sum\n");
    return 1;
  }

  if (clEnqueueWriteBuffer(command_queue, work_buffer_a, CL_TRUE, 0,
                           buffer_width * buffer_height * sizeof(cl_float), a,
                           0, NULL, NULL) != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue write of buffer a\n");
    return 1;
  }
  if (clEnqueueWriteBuffer(command_queue, work_buffer_b, CL_TRUE, 0,
                           buffer_width * buffer_height * sizeof(cl_float), b,
                           0, NULL, NULL) != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue write of buffer b\n");
    return 1;
  }

  // Read a write b
  clSetKernelArg(kernela, 0, sizeof(cl_mem), &work_buffer_a);
  clSetKernelArg(kernela, 1, sizeof(cl_mem), &work_buffer_b);
  clSetKernelArg(kernela, 2, sizeof(cl_float), &diffusion_constant);
  clSetKernelArg(kernela, 3, sizeof(int), &buffer_width);
  clSetKernelArg(kernela, 4, sizeof(int), &buffer_height);

  // Read b write a
  clSetKernelArg(kernelb, 0, sizeof(cl_mem), &work_buffer_b);
  clSetKernelArg(kernelb, 1, sizeof(cl_mem), &work_buffer_a);
  clSetKernelArg(kernelb, 2, sizeof(cl_float), &diffusion_constant);
  clSetKernelArg(kernelb, 3, sizeof(int), &buffer_width);
  clSetKernelArg(kernelb, 4, sizeof(int), &buffer_height);

  const size_t global_sz[] = {buffer_width - 2, buffer_height - 2};
  for (int i = 0; i < (iterations + 1) / 2; i++) {
    if (clEnqueueNDRangeKernel(command_queue, kernela, 2, NULL,
                               (const size_t *)&global_sz, NULL, 0, NULL,
                               NULL) != CL_SUCCESS) {
      fprintf(stderr, "cannot enqueue kernel a\n");
      return 1;
    }
    if (clEnqueueNDRangeKernel(command_queue, kernelb, 2, NULL,
                               (const size_t *)&global_sz, NULL, 0, NULL,
                               NULL) != CL_SUCCESS) {
      fprintf(stderr, "cannot enqueue kernel\n");
      return 1;
    }
  }

  // Set arguments for reduction kernel
  const cl_int sz_clint = (cl_int)sz;
  // TODO Check if correct order
  if (iterations & 1) {
    clSetKernelArg(kernel_reduction, 0, sizeof(cl_mem), &work_buffer_b);
  } else {
    clSetKernelArg(kernel_reduction, 0, sizeof(cl_mem), &work_buffer_a);
  }
  clSetKernelArg(kernel_reduction, 1, local_redsz * sizeof(float), NULL);
  clSetKernelArg(kernel_reduction, 2, sizeof(cl_int), &sz_clint);
  clSetKernelArg(kernel_reduction, 3, sizeof(cl_mem), &reduction_buffer);

  size_t global_redsz_szt = (size_t)global_redsz;
  size_t local_redsz_szt = (size_t)local_redsz;
  if (clEnqueueNDRangeKernel(command_queue, kernel_reduction, 1, NULL,
                             (const size_t *)&global_redsz_szt,
                             (const size_t *)&local_redsz_szt, 0, NULL,
                             NULL) != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue kernel reduction\n");
    return 1;
  }

  float *reduction_result = malloc(nmb_redgps * sizeof(float));
  if (clEnqueueReadBuffer(command_queue, reduction_buffer, CL_TRUE, 0,
                          nmb_redgps * sizeof(float), reduction_result, 0, NULL,
                          NULL) != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue read of buffer c\n");
    return 1;
  }

  if (clFinish(command_queue) != CL_SUCCESS) {
    fprintf(stderr, "cannot finish queue\n");
    return 1;
  }

  double reduction_total = 0;
  for (size_t ix = 0; ix < nmb_redgps; ++ix)
    reduction_total += reduction_result[ix];
  cl_float average = reduction_total / (height * width);

  if (iterations & 1) {
    clSetKernelArg(kernel_heat_difference, 0, sizeof(cl_mem), &work_buffer_b);
    clSetKernelArg(kernel_heat_difference, 1, sizeof(cl_mem), &work_buffer_a);
  } else {
    clSetKernelArg(kernel_heat_difference, 0, sizeof(cl_mem), &work_buffer_a);
    clSetKernelArg(kernel_heat_difference, 1, sizeof(cl_mem), &work_buffer_b);
  }
  clSetKernelArg(kernel_heat_difference, 2, sizeof(cl_float), &average);
  clSetKernelArg(kernel_heat_difference, 3, sizeof(int), &buffer_width);
  clSetKernelArg(kernel_heat_difference, 4, sizeof(int), &buffer_height);

  if (clEnqueueNDRangeKernel(command_queue, kernel_heat_difference, 2, NULL,
                             (const size_t *)&global_sz, NULL, 0, NULL,
                             NULL) != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue heat heat_difference kernel\n");
    return 1;
  }

  // Set correct read argument for second reduction
  if (iterations & 1) {
    clSetKernelArg(kernel_reduction, 0, sizeof(cl_mem), &work_buffer_a);
  } else {
    clSetKernelArg(kernel_reduction, 0, sizeof(cl_mem), &work_buffer_b);
  }

  if (clEnqueueNDRangeKernel(command_queue, kernel_reduction, 1, NULL,
                             (const size_t *)&global_redsz_szt,
                             (const size_t *)&local_redsz_szt, 0, NULL,
                             NULL) != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue kernel reduction\n");
    return 1;
  }

  if (clEnqueueReadBuffer(command_queue, reduction_buffer, CL_TRUE, 0,
                          nmb_redgps * sizeof(float), reduction_result, 0, NULL,
                          NULL) != CL_SUCCESS) {
    fprintf(stderr, "cannot enqueue read of buffer c\n");
    return 1;
  }

  if (clFinish(command_queue) != CL_SUCCESS) {
    fprintf(stderr, "cannot finish queue\n");
    return 1;
  }

  reduction_total = 0;
  for (size_t ix = 0; ix < nmb_redgps; ++ix)
    reduction_total += reduction_result[ix];
  cl_float difference_average = reduction_total / (height * width);

  /* If you're interested in last iteration:
  if (iterations & 1) {
    if (clEnqueueReadBuffer(command_queue, work_buffer_b, CL_TRUE, 0,
                            buffer_width * buffer_height * sizeof(cl_float), a,
                            0, NULL, NULL) != CL_SUCCESS) {
      fprintf(stderr, "cannot enqueue read of buffer a\n");
      return 1;
    }
  } else {
    if (clEnqueueReadBuffer(command_queue, work_buffer_a, CL_TRUE, 0,
                            buffer_width * buffer_height * sizeof(cl_float), a,
                            0, NULL, NULL) != CL_SUCCESS) {
      fprintf(stderr, "cannot enqueue read of buffer b\n");
      return 1;
    }
  }

  if (clFinish(command_queue) != CL_SUCCESS) {
    fprintf(stderr, "cannot finish queue\n");
    return 1;
  }

    for (size_t jx = 1; jx < buffer_height; ++jx) {
      for (size_t ix = 1; ix < buffer_width; ++ix) {
        printf(" %5.f ", a[jx * buffer_width + ix]);
        printf("\n");
      }
    }
  */
  printf("Average: %f\n", average);
  printf("Average of differences: %f\n", difference_average);

  // Free objects
  free(a);
  free(b);

  clReleaseMemObject(work_buffer_a);
  clReleaseMemObject(work_buffer_b);

  clReleaseProgram(program);
  clReleaseKernel(kernela);
  clReleaseKernel(kernelb);

  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
