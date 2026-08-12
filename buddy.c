#include "buddy.h"
#include <stddef.h>
#include <stdlib.h>

#define PAGE_SIZE 4096
#define MAX_RANK 16

/* State values per page index:
 * 0: page is inside a larger block (not a block start)
 * 1: page is the start of a free block
 * 2: page is the start of an allocated block
 */
static void *g_base = NULL;
static int g_total_pages = 0;

static unsigned char *g_block_rank = NULL; /* rank of the block containing page i */
static unsigned char *g_state = NULL;      /* 0/1/2, see above */
static int *g_free_next = NULL;            /* next free block of same rank */
static int *g_free_prev = NULL;            /* previous free block of same rank */
static int g_free_head[MAX_RANK + 2];
static int g_free_count[MAX_RANK + 2];

static void cleanup(void) {
    if (g_block_rank) { free(g_block_rank); g_block_rank = NULL; }
    if (g_state) { free(g_state); g_state = NULL; }
    if (g_free_next) { free(g_free_next); g_free_next = NULL; }
    if (g_free_prev) { free(g_free_prev); g_free_prev = NULL; }
}

static void add_free(int idx, int rank) {
    g_free_next[idx] = g_free_head[rank];
    if (g_free_head[rank] != -1) {
        g_free_prev[g_free_head[rank]] = idx;
    }
    g_free_prev[idx] = -1;
    g_free_head[rank] = idx;
    g_free_count[rank]++;
}

static void remove_free(int idx, int rank) {
    if (g_free_prev[idx] != -1) {
        g_free_next[g_free_prev[idx]] = g_free_next[idx];
    } else {
        g_free_head[rank] = g_free_next[idx];
    }
    if (g_free_next[idx] != -1) {
        g_free_prev[g_free_next[idx]] = g_free_prev[idx];
    }
    g_free_count[rank]--;
}

int init_page(void *p, int pgcount) {
    if (p == NULL || pgcount <= 0) {
        return -EINVAL;
    }

    cleanup();

    g_base = p;
    g_total_pages = pgcount;

    g_block_rank = (unsigned char *)calloc(pgcount, sizeof(unsigned char));
    g_state = (unsigned char *)calloc(pgcount, sizeof(unsigned char));
    g_free_next = (int *)malloc(pgcount * sizeof(int));
    g_free_prev = (int *)malloc(pgcount * sizeof(int));

    if (g_block_rank == NULL || g_state == NULL ||
        g_free_next == NULL || g_free_prev == NULL) {
        cleanup();
        return -ENOSPC;
    }

    for (int i = 0; i <= MAX_RANK; i++) {
        g_free_head[i] = -1;
        g_free_count[i] = 0;
    }

    for (int i = 0; i < pgcount; i++) {
        g_free_next[i] = -1;
        g_free_prev[i] = -1;
    }

    /* Greedy decomposition of pgcount into power-of-2 blocks.
     * Covers all pages, even when pgcount is not a power of 2.
     */
    int start = 0;
    int remaining = pgcount;
    while (remaining > 0) {
        int rank = 1;
        int size = 1;
        while ((size << 1) <= remaining && rank < MAX_RANK) {
            size <<= 1;
            rank++;
        }
        for (int i = start; i < start + size; i++) {
            g_block_rank[i] = (unsigned char)rank;
        }
        g_state[start] = 1;
        add_free(start, rank);
        start += size;
        remaining -= size;
    }

    return OK;
}

void *alloc_pages(int rank) {
    if (rank < 1 || rank > MAX_RANK) {
        return ERR_PTR(-EINVAL);
    }

    /* Find the smallest free block that is large enough. */
    int k = rank;
    while (k <= MAX_RANK && g_free_head[k] == -1) {
        k++;
    }
    if (k > MAX_RANK) {
        return ERR_PTR(-ENOSPC);
    }

    int idx = g_free_head[k];
    remove_free(idx, k);

    /* Split down to the requested rank. */
    while (k > rank) {
        int half = 1 << (k - 2);          /* pages in each half */
        int right = idx + half;            /* start of right half */
        int new_rank = k - 1;
        int end = idx + (half << 1);       /* one past end of block */
        for (int i = idx; i < end; i++) {
            g_block_rank[i] = (unsigned char)new_rank;
        }
        g_state[right] = 1;                /* right half becomes a free block */
        add_free(right, new_rank);
        k = new_rank;
    }

    g_state[idx] = 2;                      /* allocated block start */
    return (char *)g_base + (long)idx * PAGE_SIZE;
}

int return_pages(void *p) {
    if (p == NULL || g_base == NULL) {
        return -EINVAL;
    }
    long offset = (char *)p - (char *)g_base;
    if (offset < 0 || offset % PAGE_SIZE != 0) {
        return -EINVAL;
    }
    int idx = (int)(offset / PAGE_SIZE);
    if (idx < 0 || idx >= g_total_pages) {
        return -EINVAL;
    }
    if (g_state[idx] != 2) {
        return -EINVAL;
    }
    int rank = g_block_rank[idx];
    if (rank < 1 || rank > MAX_RANK) {
        return -EINVAL;
    }

    g_state[idx] = 0;
    int cur_idx = idx;
    int cur_rank = rank;

    /* Coalesce with buddy while possible. */
    while (cur_rank < MAX_RANK) {
        int buddy = cur_idx ^ (1 << (cur_rank - 1));
        if (buddy < 0 || buddy >= g_total_pages) {
            break;
        }
        if (g_state[buddy] != 1) {
            break;
        }
        if (g_block_rank[buddy] != cur_rank) {
            break;
        }

        remove_free(buddy, cur_rank);
        g_state[buddy] = 0;

        int new_idx = (buddy < cur_idx) ? buddy : cur_idx;
        int new_rank = cur_rank + 1;
        int size = 1 << (new_rank - 1);
        for (int i = new_idx; i < new_idx + size && i < g_total_pages; i++) {
            g_block_rank[i] = (unsigned char)new_rank;
        }
        cur_idx = new_idx;
        cur_rank = new_rank;
    }

    g_state[cur_idx] = 1;
    add_free(cur_idx, cur_rank);
    return OK;
}

int query_ranks(void *p) {
    if (p == NULL || g_base == NULL) {
        return -EINVAL;
    }
    long offset = (char *)p - (char *)g_base;
    if (offset < 0 || offset % PAGE_SIZE != 0) {
        return -EINVAL;
    }
    int idx = (int)(offset / PAGE_SIZE);
    if (idx < 0 || idx >= g_total_pages) {
        return -EINVAL;
    }
    if (g_block_rank[idx] == 0) {
        return -EINVAL;
    }
    return g_block_rank[idx];
}

int query_page_counts(int rank) {
    if (rank < 1 || rank > MAX_RANK) {
        return -EINVAL;
    }
    return g_free_count[rank];
}
