
/** 
 * Decompress CUDA Fatbinary compiled with -Xfatbin -compress-all 
 * 
 * @note: This is a third-party module not related with Neutrino Project
 * $ gcc -O3 -o decompress decompress.c # compile
 * $ ./decompress <input_file> <output_file> # use
 */

/** Original Author Information
 *
 * Author: Niklas Eiling <niklas.eiling@rwth-aachen.de>
 * SPDX-FileCopyrightText: 2023 Niklas Eiling <niklas.eiling@rwth-aachen.de>
 * SPDX-License-Identifier: Apache-2.0
 *********************************************************************************/

#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

struct  __attribute__((__packed__)) fat_elf_header
{
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t size;
};

struct  __attribute__((__packed__)) fat_text_header
{
    uint16_t kind;
    uint16_t unknown1;
    uint32_t header_size;
    uint64_t size;
    uint32_t compressed_size;       // Size of compressed data
    uint32_t unknown2;              // Address size for PTX?
    uint16_t minor;
    uint16_t major;
    uint32_t arch;
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;                  // Alignment for compression?
    uint64_t decompressed_size;     // Length of compressed data in decompressed representation.
                                    // There is an uncompressed footer so this is generally smaller
                                    // than size.
};

#define FATBIN_TEXT_MAGIC     0xBA55ED50
#define FATBIN_FLAG_64BIT     0x0000000000000001LL
#define FATBIN_FLAG_DEBUG     0x0000000000000002LL
#define FATBIN_FLAG_LINUX     0x0000000000000010LL
#define FATBIN_FLAG_COMPRESS  0x0000000000002000LL

static void print_header(struct fat_text_header *th)
{
    char* flagstr = NULL;
    asprintf(&flagstr, "64Bit: %s, Debug: %s, Linux: %s, Compress: %s",
        (th->flags & FATBIN_FLAG_64BIT) ? "yes" : "no",
        (th->flags & FATBIN_FLAG_DEBUG) ? "yes" : "no",
        (th->flags & FATBIN_FLAG_LINUX) ? "yes" : "no",
        (th->flags & FATBIN_FLAG_COMPRESS) ? "yes" : "no");

    printf("text_header: fatbin_kind: %#x, header_size %#x, size %#zx, compressed_size %#x,\
 minor %#x, major %#x, arch %d, decompressed_size %#zx\n\tflags: %s\n",
        th->kind,
        th->header_size,
        th->size,
        th->compressed_size,
        th->minor,
        th->major,
        th->arch,
        th->decompressed_size,
        flagstr);
}

static int get_elf_header(const uint8_t* fatbin_data, size_t fatbin_size, struct fat_elf_header **elf_header)
{
    struct fat_elf_header *eh = NULL;

    if (fatbin_data == NULL || elf_header == NULL) {
        fprintf(stderr, "Error: fatbin_data is NULL\n");
        return 1;
    }

    if (fatbin_size < sizeof(struct fat_elf_header)) {
        fprintf(stderr, "Error: fatbin_size is too small\n");
        return 1;
    }

    eh = (struct fat_elf_header*) fatbin_data;
    if (eh->magic != FATBIN_TEXT_MAGIC) {
        fprintf(stderr, "Error: Invalid magic  number: expected %#x but got %#x\n", FATBIN_TEXT_MAGIC, eh->magic);
        return 1;
    }

    if (eh->version != 1 || eh->header_size != sizeof(struct fat_elf_header)) {
        fprintf(stderr, "fatbin text version is wrong or header size is inconsistent.\
            This is a sanity check to avoid reading a new fatbinary format\n");
        return 1;
    }
    *elf_header = eh;
    return 0;
}

static int get_text_header(const uint8_t* fatbin_data, size_t fatbin_size, struct fat_text_header **text_header)
{
    struct fat_text_header *th = NULL;

    if (fatbin_data == NULL || text_header == NULL) {
        fprintf(stderr, "Error: fatbin_data is NULL\n");
        return 1;
    }

    if (fatbin_size < sizeof(struct fat_text_header)) {
        fprintf(stderr, "Error: fatbin_size is too small\n");
        return 1;
    }

    th = (struct fat_text_header*)fatbin_data;

    if(th->obj_name_offset != 0) {
        if (((char*)th)[th->obj_name_offset + th->obj_name_len] != '\0') {
            printf("Fatbin object name is not null terminated\n");
        } else {
            char *obj_name = (char*)th + th->obj_name_offset;
            printf("Fatbin object name: %s (len:%#x)\n", obj_name, th->obj_name_len);
        }
    }

    *text_header = th;
    return 0;
}

size_t decompress(const uint8_t* input, size_t input_size, uint8_t* output, size_t output_size)
{
    size_t ipos = 0, opos = 0;  
    uint64_t next_nclen;  // length of next non-compressed segment
    uint64_t next_clen;   // length of next compressed segment
    uint64_t back_offset; // negative offset where redudant data is located, relative to current opos

    while (ipos < input_size) {
        next_nclen = (input[ipos] & 0xf0) >> 4;
        next_clen = 4 + (input[ipos] & 0xf);
        if (next_nclen == 0xf) {
            do {
                next_nclen += input[++ipos];
            } while (input[ipos] == 0xff);
        }
        
        if (memcpy(output + opos, input + (++ipos), next_nclen) == NULL) {
            fprintf(stderr, "Error copying data");
            return 0;
        }

        ipos += next_nclen;
        opos += next_nclen;
        if (ipos >= input_size || opos >= output_size) {
            break;
        }
        back_offset = input[ipos] + (input[ipos + 1] << 8);
        ipos += 2;
        if (next_clen == 0xf+4) {
            do {
                next_clen += input[ipos++];
            } while (input[ipos - 1] == 0xff);
        }

        if (next_clen <= back_offset) {
            if (memcpy(output + opos, output + opos - back_offset, next_clen) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
        } else {
            if (memcpy(output + opos, output + opos - back_offset, back_offset) == NULL) {
                fprintf(stderr, "Error copying data");
                return 0;
            }
            for (size_t i = back_offset; i < next_clen; i++) {
                output[opos + i] = output[opos + i - back_offset];
            }
        }

        opos += next_clen;
    }
    return opos;
}

int decompress_section(const uint8_t *input, uint8_t **output, size_t *output_size,
                       struct fat_elf_header *eh, struct fat_text_header *th, size_t *eh_out_offset,
                       size_t *input_read)
{
    struct fat_text_header *th_out = NULL;
    struct fat_elf_header *eh_out = NULL;
    uint8_t *output_pos = 0;
    size_t padding;
    int ret = 0;
    const uint8_t zeroes[6] = {0};

    if (output == NULL || output_size == NULL || eh == NULL || th == NULL || eh_out_offset == NULL || input_read == NULL) {
        fprintf(stderr, "Error: invalid parameters\n");
        return -1;
    }
    *input_read = 0;

    // reallocate the output memory region
    if ((*output = realloc(*output, *output_size + th->decompressed_size + eh->header_size + th->header_size)) == NULL) {
        fprintf(stderr, "Error allocating memory of size %#zx for output buffer: %s\n", 
                *output_size + th->decompressed_size + eh->header_size + th->header_size, strerror(errno));
        ret = -1;
        goto error;
    }
    output_pos = *output + *output_size;
    *output_size += th->decompressed_size + th->header_size;

    if (input == (uint8_t*)eh + eh->header_size + th->header_size) { // We are at the first section
        if (memcpy(output_pos, eh, eh->header_size) == NULL) {
            fprintf(stderr, "Error copying data");
            ret = -1;
            goto error;
        }
        eh_out = ((struct fat_elf_header*)(output_pos));
        eh_out->size = 0;
        *eh_out_offset = output_pos - *output;
        output_pos += eh->header_size;
        *output_size += eh->header_size;
    }
    eh_out = ((struct fat_elf_header*)(*output + *eh_out_offset)); // repair pointer in case realloc moved the buffer
    eh_out->size += th->decompressed_size + th->header_size;       // set size

    if (memcpy(output_pos, th, th->header_size) == NULL) {
        fprintf(stderr, "Error copying data");
        ret = -1;
        goto error;
    }
    th_out = ((struct fat_text_header*)output_pos);
    th_out->flags &= ~FATBIN_FLAG_COMPRESS;  // clear compressed flag
    th_out->compressed_size = 0;             // clear compressed size
    th_out->decompressed_size = 0;           // clear decompressed size
    th_out->size = th->decompressed_size;    // set size

    output_pos += th->header_size;

    size_t decompress_ret;

    if ((decompress_ret = decompress(input, th->compressed_size, output_pos, th->decompressed_size)) != th->decompressed_size) {
        fprintf(stderr, "Decompression failed: decompressed size (%#0zx) is not as indicated in header (%#0zx).\n",
                decompress_ret, th->decompressed_size);
        ret = -1;
    }

    *input_read += th->compressed_size;
    output_pos += th->decompressed_size;
    
    padding = ((8 - (size_t)(input + *input_read)) % 8);
    if (memcmp(input + *input_read, zeroes, padding) != 0) {
        printf("Error: expected %#zx zero bytes, got:\n", padding);
        goto error;
    }
    input_read += padding;

    padding = ((8 - (size_t)th->decompressed_size) % 8);
    // Because we always allocated enough memory for one more elf_header and this is smaller than
    // the maximal padding of 7, we do not have to reallocate here.
    memset(output_pos, 0, padding);
    *output_size += padding;
    eh_out->size += padding;
    th_out->size += padding;

    return ret;
 error:
    free(*output);
    *output = NULL;
    return ret;
}

/** Decompresses a fatbin file
 * @param fatbin_data Pointer to the fatbin data
 * @param fatbin_size Size of the fatbin data
 * @param decompressed_data Pointer to a variable that will be set to point to the decompressed data
 * @param decompressed_size Pointer to a variable that will be set to the size of the decompressed data
 */
size_t decompress_fatbin(const uint8_t* fatbin_data, size_t fatbin_size, uint8_t** decompressed_data)
{
    struct fat_elf_header *eh = NULL;
    size_t eh_out_offset = 0;
    struct fat_text_header *th = NULL;
    const uint8_t *input_pos = fatbin_data;

    uint8_t *output = NULL;
    size_t output_size = 0;
    size_t input_read;

    if (fatbin_data == NULL || decompressed_data == NULL) {
        fprintf(stderr, "Error: fatbin_data is NULL\n");
        goto error;
    }
    int i = 1;

    while (input_pos < fatbin_data + fatbin_size) {
        if (get_elf_header(input_pos, fatbin_size - (input_pos - fatbin_data), &eh) != 0) {
            fprintf(stderr, "Something went wrong while checking the header.\n");
            goto error;
        }
        printf("elf header no. %d: magic: %#x, version: %#x, header_size: %#x, size: %#zx\n",
               i++, eh->magic, eh->version, eh->header_size, eh->size);
        input_pos += eh->header_size;
        do {
            if (get_text_header(input_pos, fatbin_size - (input_pos - fatbin_data), &th) != 0) {
                fprintf(stderr, "Something went wrong while checking the header.\n");
                goto error;
            }
            print_header(th);
            if (th->decompressed_size == 0) {
                fprintf(stderr, "Error: decompressed size is 0.\n");
                goto soft_error;
            }
            input_pos += th->header_size;

            if (decompress_section(input_pos, &output, &output_size, eh, th, &eh_out_offset, &input_read) != 0) {
                fprintf(stderr, "Something went wrong while decompressing text section.\n");
                goto soft_error;
            }

            // @bugfix don't trust input_read, instead, uses the th->size
            input_pos += th->size; 

        } while (input_pos < (uint8_t*)eh + (uint64_t)(eh->header_size) + eh->size);
    }
 soft_error:
    *decompressed_data = output;
    return output_size;
 error:
    if (output != NULL) {
        free(output);
    }
    *decompressed_data = NULL;
    return 0;
}

struct mapped_file {
    int fd;
    uint8_t *data;
    size_t size;
};

int mf_open_file(const char *filename, struct mapped_file *mf)
{
    struct stat st;

    if (filename == NULL || mf == NULL) {
        fprintf(stderr, "Invalid arguments\n");
        return 1;
    }

    if ((mf->fd = open(filename, O_RDONLY)) == -1) {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        return 1;
    }

    if (fstat(mf->fd, &st) == -1) {
        fprintf(stderr, "Error getting file size: %s\n", strerror(errno));
        return 1;
    }

    mf->size = st.st_size;

    if ((mf->data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, mf->fd, 0)) == MAP_FAILED) {
        fprintf(stderr, "Error mapping file: %s\n", strerror(errno));
        return 1;
    }
    return 0;
}

void mf_close(struct mapped_file *mf)
{
    if (mf == NULL) {
        return;
    }

    if (mf->data != NULL) {
        munmap(mf->data, mf->size);
    }
    mf->data = NULL;
    mf->size = 0;

    if (mf->fd != -1) {
        close(mf->fd);
    }
}

int main(int argc, char *argv[])
{
    struct mapped_file mf;
    uint8_t *output = NULL; // 
    size_t output_size = 0; // 

    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    if (mf_open_file(argv[1], &mf) != 0) {
        fprintf(stderr, "Error opening mapped file: %s\n", strerror(errno));
        return 1;
    }

    printf("Compressed file size: %#0zx\n", mf.size);

    if ((output_size = decompress_fatbin(mf.data, mf.size, &output)) == 0) {
        fprintf(stderr, "Error decompressing fatbin\n");
        return 1;
    }

    printf("Decompressed data size: %#0zx\n", output_size);
    FILE *fp = fopen(argv[2], "wb");
    fwrite(output, output_size, 1, fp);
    fclose(fp);

    mf_close(&mf);
    free(output);
    printf("success.\n");
    return 0;
}
