__kernel
void
heat_diffusion(
    __global float *a,
    __global float *b,
    float c,
    int width,
    int height
    )
{
  int ix = 1 + get_global_id(0);
  int iy = 1 + get_global_id(1);
  int center = iy * width + ix;
  int top = (iy-1)*width + ix;
  int bottom = (iy + 1) * width + ix;
  int left = iy * width + ix-1;
  int right = iy * width + ix + 1;
  
  float newvalue = a[center] + c*((a[left]+a[right]+a[top]+a[bottom])/4.0f - a[center]); 
  b[center] = newvalue;

}

__kernel
void
heat_difference(
    __global float *a,
    __global float *b,
    float c,
    int width,
    int height
    )
{
  int ix = 1 + get_global_id(0);
  int iy = 1 + get_global_id(1);
  int center = iy * width + ix;

  float newvalue = fabs(a[center]- c);
  b[center] = newvalue;

}



__kernel
void
reduction(
  __global float *c,
  __local float *scratch,
  __const int sz,
  __global float *result
  )
{
  int gsz = get_global_size(0);
  int gix = get_global_id(0);
  int lsz = get_local_size(0);
  int lix = get_local_id(0);

  float acc = 0;
  for ( int cix = get_global_id(0); cix < sz; cix += gsz )
    acc += c[cix];

  scratch[lix] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(int offset = lsz/2; offset > 0; offset /= 2) {
    if ( lix < offset )
      scratch[lix] += scratch[lix+offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if ( lix == 0 )
    result[get_group_id(0)] = scratch[0];
}
