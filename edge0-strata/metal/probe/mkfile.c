// mkfile.c - write a dense test file of N GiB with F_NOCACHE (so it is not left in the page cache):
// the first uint32 of every 4 KiB is (offset >> 12) ^ 0x5a5a5a5a, the rest is a cheap pattern.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char** argv) {
    if (argc < 3) return 1;
    const uint64_t gib = strtoull(argv[2], 0, 10);
    int fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 2;
    fcntl(fd, F_NOCACHE, 1);
    const size_t B = 64u << 20;
    uint32_t* buf = aligned_alloc(16384, B);
    for (uint64_t off = 0; off < (gib << 30); off += B) {
        for (size_t i = 0; i < B / 4; ++i) buf[i] = (uint32_t) (i * 2654435761u);
        for (size_t p = 0; p < B; p += 4096) buf[p / 4] = (uint32_t) ((off + p) >> 12) ^ 0x5a5a5a5au;
        if (write(fd, buf, B) != (ssize_t) B) return 3;
    }
    close(fd);
    return 0;
}
