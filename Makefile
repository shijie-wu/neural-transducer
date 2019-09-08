all: src/libalign.so

src/libalign.so: src/align.c
	gcc -O3 -Wall -Wextra -shared -fPIC src/align.c -o src/libalign.so

clean:
	/bin/rm src/libalign.so
