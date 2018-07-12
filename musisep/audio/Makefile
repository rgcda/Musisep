CFLAGS = -I/usr/include/python3.5 -shared -pthread -fPIC -fwrapv -O2 \
  -fno-strict-aliasing -fopenmp

all: specttool.so

specttool.c specttool.html: specttool.pyx
	cython3 -a specttool.pyx

specttool.so: specttool.c
	gcc $(CFLAGS) -o specttool.so specttool.c

clean:
	rm specttool.c specttool.html specttool.so

