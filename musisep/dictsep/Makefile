CFLAGS = -I/usr/include/python3.5 -shared -pthread -fPIC -fwrapv -O2 \
  -fno-strict-aliasing -fopenmp

all: exptool.so

exptool.c exptool.html: exptool.pyx
	cython3 -a exptool.pyx

exptool.so: exptool.c
	gcc $(CFLAGS) -o exptool.so exptool.c

clean:
	rm exptool.c exptool.html exptool.so
