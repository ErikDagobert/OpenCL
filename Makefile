BINS = diffusion 
CFLAGS = -O2 -lOpenCL

.PHONY : all
all : $(BINS) 

diffusion : diffusion.c
	gcc $(CFLAGS) -o $@ $<

archive : diffusion.c diffusion.cl Makefile
	tar czf hpcgp49_opencl.tar.gz $^

.PHONY : clean
clean :
	rm -rf $(BINS)
