#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef void liq_attr;
typedef void liq_image;
typedef void liq_result;

typedef struct {
    uint8_t r, g, b, a;
} liq_color;

typedef struct {
    unsigned int count;
    liq_color entries[256];
} liq_palette;

typedef struct {
    void *handle;
    liq_attr *(*attr_create)(void);
    void (*attr_destroy)(liq_attr *);
    int (*set_max_colors)(liq_attr *, int);
    int (*set_quality)(liq_attr *, int, int);
    int (*set_speed)(liq_attr *, int);
    liq_image *(*image_create_rgba)(liq_attr *, const void *, int, int, double);
    void (*image_destroy)(liq_image *);
    int (*image_quantize)(liq_image *, liq_attr *, liq_result **);
    int (*set_dithering_level)(liq_result *, float);
    int (*write_remapped_image)(liq_result *, liq_image *, void *, size_t);
    const liq_palette *(*get_palette)(liq_result *);
    void (*result_destroy)(liq_result *);
} liq_api;

typedef struct {
    uint32_t packed;
    uint64_t count;
    double lab[3];
} observed_candidate;

typedef struct {
    size_t index;
    double distance;
} ranked_candidate;

typedef struct {
    liq_color color;
    uint64_t count;
    unsigned int original_index;
} palette_cluster;

typedef struct {
    const liq_api *api;
    const uint8_t *rgb;
    const size_t *offsets;
    size_t box_count;
    int speed;
    int observed;
    double min_cluster_ratio;
    _Atomic size_t next_box;
    _Atomic int status;
    uint8_t *palettes;
    uint64_t *counts;
    double *weights;
    uint8_t *sizes;
} batch_context;

static int compare_u32(const void *left, const void *right) {
    uint32_t a = *(const uint32_t *)left;
    uint32_t b = *(const uint32_t *)right;
    return (a > b) - (a < b);
}

static int compare_ranked(const void *left, const void *right) {
    const ranked_candidate *a = (const ranked_candidate *)left;
    const ranked_candidate *b = (const ranked_candidate *)right;
    if (a->distance < b->distance) return -1;
    if (a->distance > b->distance) return 1;
    return (a->index > b->index) - (a->index < b->index);
}

static double lab_distance(const double left[3], const double right[3]);

static void ranked_sift_down(ranked_candidate *heap, size_t size, size_t root) {
    for (;;) {
        size_t child = root * 2 + 1;
        if (child >= size) return;
        if (child + 1 < size && compare_ranked(&heap[child], &heap[child + 1]) < 0) {
            child++;
        }
        if (compare_ranked(&heap[root], &heap[child]) >= 0) return;
        ranked_candidate temporary = heap[root];
        heap[root] = heap[child];
        heap[child] = temporary;
        root = child;
    }
}

static size_t select_nearest(
    ranked_candidate *heap,
    const observed_candidate *candidates,
    size_t candidate_count,
    const double target_lab[3]
) {
    size_t heap_size = candidate_count < 64 ? candidate_count : 64;
    for (size_t index = 0; index < heap_size; index++) {
        heap[index].index = index;
        heap[index].distance = lab_distance(target_lab, candidates[index].lab);
    }
    for (size_t index = heap_size / 2; index > 0; index--) {
        ranked_sift_down(heap, heap_size, index - 1);
    }
    for (size_t index = heap_size; index < candidate_count; index++) {
        ranked_candidate candidate = {
            .index = index,
            .distance = lab_distance(target_lab, candidates[index].lab),
        };
        if (compare_ranked(&candidate, &heap[0]) < 0) {
            heap[0] = candidate;
            ranked_sift_down(heap, heap_size, 0);
        }
    }
    qsort(heap, heap_size, sizeof(ranked_candidate), compare_ranked);
    return heap_size;
}

static int compare_clusters(const void *left, const void *right) {
    const palette_cluster *a = (const palette_cluster *)left;
    const palette_cluster *b = (const palette_cluster *)right;
    if (a->count > b->count) return -1;
    if (a->count < b->count) return 1;
    return (a->original_index > b->original_index) -
           (a->original_index < b->original_index);
}

static double srgb_linear(uint8_t channel) {
    double value = (double)channel / 255.0;
    return value <= 0.04045
        ? value / 12.92
        : pow((value + 0.055) / 1.055, 2.4);
}

static void rgb_to_oklab(uint32_t packed, double output[3]) {
    double red = srgb_linear((uint8_t)(packed >> 16));
    double green = srgb_linear((uint8_t)(packed >> 8));
    double blue = srgb_linear((uint8_t)packed);
    double light = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue;
    double medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue;
    double short_value = 0.0883024619 * red + 0.2817188374 * green + 0.6299787005 * blue;
    double light_root = cbrt(light);
    double medium_root = cbrt(medium);
    double short_root = cbrt(short_value);
    output[0] = 0.2104542553 * light_root + 0.7936177850 * medium_root - 0.0040720468 * short_root;
    output[1] = 1.9779984951 * light_root - 2.4285922050 * medium_root + 0.4505937099 * short_root;
    output[2] = 0.0259040371 * light_root + 0.7827717662 * medium_root - 0.8086757660 * short_root;
}

static double lab_distance(const double left[3], const double right[3]) {
    double dl = left[0] - right[0];
    double da = left[1] - right[1];
    double db = left[2] - right[2];
    return sqrt(dl * dl + da * da + db * db);
}

#define LOAD_SYMBOL(api, field, symbol) do { \
    *(void **)(&(api)->field) = dlsym((api)->handle, symbol); \
    if (!(api)->field) return -2; \
} while (0)

static int load_api(const char *path, liq_api *api) {
    api->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!api->handle) return -1;
    LOAD_SYMBOL(api, attr_create, "liq_attr_create");
    LOAD_SYMBOL(api, attr_destroy, "liq_attr_destroy");
    LOAD_SYMBOL(api, set_max_colors, "liq_set_max_colors");
    LOAD_SYMBOL(api, set_quality, "liq_set_quality");
    LOAD_SYMBOL(api, set_speed, "liq_set_speed");
    LOAD_SYMBOL(api, image_create_rgba, "liq_image_create_rgba");
    LOAD_SYMBOL(api, image_destroy, "liq_image_destroy");
    LOAD_SYMBOL(api, image_quantize, "liq_image_quantize");
    LOAD_SYMBOL(api, set_dithering_level, "liq_set_dithering_level");
    LOAD_SYMBOL(api, write_remapped_image, "liq_write_remapped_image");
    LOAD_SYMBOL(api, get_palette, "liq_get_palette");
    LOAD_SYMBOL(api, result_destroy, "liq_result_destroy");
    return 0;
}

static void set_error(batch_context *context, int code) {
    int expected = 0;
    atomic_compare_exchange_strong(&context->status, &expected, code);
}

static int write_observed_palette(
    batch_context *context,
    size_t box,
    const uint8_t *source,
    size_t pixel_count,
    const liq_palette *palette,
    unsigned int palette_size,
    const uint64_t cluster_counts[5],
    uint32_t *packed,
    observed_candidate *candidates,
    ranked_candidate *ranked
) {
    for (size_t index = 0; index < pixel_count; index++) {
        packed[index] = ((uint32_t)source[index * 3] << 16) |
                        ((uint32_t)source[index * 3 + 1] << 8) |
                        (uint32_t)source[index * 3 + 2];
    }
    qsort(packed, pixel_count, sizeof(uint32_t), compare_u32);
    size_t candidate_count = 0;
    uint64_t max_frequency = 0;
    for (size_t index = 0; index < pixel_count;) {
        size_t next = index + 1;
        while (next < pixel_count && packed[next] == packed[index]) next++;
        uint64_t count = (uint64_t)(next - index);
        candidates[candidate_count].packed = packed[index];
        candidates[candidate_count].count = count;
        rgb_to_oklab(packed[index], candidates[candidate_count].lab);
        if (count > max_frequency) max_frequency = count;
        candidate_count++;
        index = next;
    }

    palette_cluster clusters[5];
    uint64_t total = 0;
    for (unsigned int index = 0; index < palette_size; index++) {
        clusters[index].color = palette->entries[index];
        clusters[index].count = cluster_counts[index];
        clusters[index].original_index = index;
        total += cluster_counts[index];
    }
    qsort(clusters, palette_size, sizeof(palette_cluster), compare_clusters);
    unsigned int kept[5];
    unsigned int kept_count = 0;
    for (unsigned int index = 0; index < palette_size; index++) {
        double ratio = total ? (double)clusters[index].count / (double)total : 0.0;
        if (ratio >= context->min_cluster_ratio) kept[kept_count++] = index;
    }
    if (kept_count == 0 && palette_size > 0) kept[kept_count++] = 0;

    size_t selected[5];
    unsigned int selected_count = 0;
    for (unsigned int output_index = 0; output_index < kept_count; output_index++) {
        palette_cluster *cluster = &clusters[kept[output_index]];
        uint32_t target_packed = ((uint32_t)cluster->color.r << 16) |
                                 ((uint32_t)cluster->color.g << 8) |
                                 (uint32_t)cluster->color.b;
        double target_lab[3];
        rgb_to_oklab(target_packed, target_lab);
        size_t neighbor_count = select_nearest(
            ranked, candidates, candidate_count, target_lab
        );
        size_t winner = (size_t)-1;
        double winning_score = INFINITY;
        for (size_t rank = 0; rank < neighbor_count; rank++) {
            size_t candidate_index = ranked[rank].index;
            int already_selected = 0;
            for (unsigned int prior = 0; prior < selected_count; prior++) {
                if (selected[prior] == candidate_index) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected) continue;
            double frequency = 0.25 + 0.75 * sqrt(
                (double)candidates[candidate_index].count / (double)max_frequency
            );
            double score = ranked[rank].distance / frequency;
            if (score < winning_score ||
                (score == winning_score && candidate_index < winner)) {
                winning_score = score;
                winner = candidate_index;
            }
        }
        if (winner == (size_t)-1) return -20;
        selected[selected_count++] = winner;
        uint32_t color = candidates[winner].packed;
        size_t output = (box * 5 + output_index) * 3;
        context->palettes[output] = (uint8_t)(color >> 16);
        context->palettes[output + 1] = (uint8_t)(color >> 8);
        context->palettes[output + 2] = (uint8_t)color;
        context->weights[box * 5 + output_index] = total
            ? (double)cluster->count / (double)total
            : 0.0;
    }
    context->sizes[box] = (uint8_t)kept_count;
    return 0;
}

static void *batch_worker(void *opaque) {
    batch_context *context = (batch_context *)opaque;
    const liq_api *api = context->api;
    liq_attr *attribute = api->attr_create();
    if (!attribute) {
        set_error(context, -3);
        return NULL;
    }
    if (api->set_max_colors(attribute, 5) != 0 ||
        api->set_quality(attribute, 0, 100) != 0 ||
        api->set_speed(attribute, context->speed) != 0) {
        api->attr_destroy(attribute);
        set_error(context, -4);
        return NULL;
    }
    uint8_t *rgba = NULL;
    uint8_t *indexed = NULL;
    uint32_t *packed = NULL;
    observed_candidate *candidates = NULL;
    ranked_candidate *ranked = NULL;
    size_t capacity = 0;

    for (;;) {
        size_t box = atomic_fetch_add(&context->next_box, 1);
        if (box >= context->box_count) break;
        size_t start = context->offsets[box];
        size_t pixel_count = context->offsets[box + 1] - start;
        if (pixel_count > capacity) {
            free(rgba);
            free(indexed);
            free(packed);
            free(candidates);
            free(ranked);
            rgba = (uint8_t *)malloc(pixel_count * 4);
            indexed = (uint8_t *)malloc(pixel_count);
            packed = context->observed
                ? (uint32_t *)malloc(pixel_count * sizeof(uint32_t)) : NULL;
            candidates = context->observed
                ? (observed_candidate *)malloc(pixel_count * sizeof(observed_candidate)) : NULL;
            ranked = context->observed
                ? (ranked_candidate *)malloc(pixel_count * sizeof(ranked_candidate)) : NULL;
            if (!rgba || !indexed ||
                (context->observed && (!packed || !candidates || !ranked))) {
                free(rgba);
                free(indexed);
                free(packed);
                free(candidates);
                free(ranked);
                rgba = NULL;
                indexed = NULL;
                packed = NULL;
                candidates = NULL;
                ranked = NULL;
                capacity = 0;
                set_error(context, -5);
                continue;
            }
            capacity = pixel_count;
        }
        const uint8_t *source = context->rgb + start * 3;
        for (size_t index = 0; index < pixel_count; index++) {
            rgba[index * 4] = source[index * 3];
            rgba[index * 4 + 1] = source[index * 3 + 1];
            rgba[index * 4 + 2] = source[index * 3 + 2];
            rgba[index * 4 + 3] = 255;
        }

        liq_image *image = api->image_create_rgba(
            attribute, rgba, (int)pixel_count, 1, 0.0
        );
        liq_result *result = NULL;
        int code = image ? api->image_quantize(image, attribute, &result) : -1;
        if (code == 0 && result) {
            api->set_dithering_level(result, 0.0f);
            code = api->write_remapped_image(
                result, image, indexed, pixel_count
            );
        }
        if (code == 0 && result) {
            const liq_palette *palette = api->get_palette(result);
            unsigned int palette_size = palette->count < 5 ? palette->count : 5;
            uint64_t cluster_counts[5] = {0, 0, 0, 0, 0};
            for (size_t index = 0; index < pixel_count; index++) {
                uint8_t color = indexed[index];
                if (color < palette_size) {
                    cluster_counts[color]++;
                }
            }
            if (context->observed) {
                int observed_code = write_observed_palette(
                    context, box, source, pixel_count, palette, palette_size,
                    cluster_counts, packed, candidates, ranked
                );
                if (observed_code != 0) set_error(context, observed_code);
            } else {
                context->sizes[box] = (uint8_t)palette_size;
                for (unsigned int color = 0; color < palette_size; color++) {
                    size_t output = (box * 5 + color) * 3;
                    context->palettes[output] = palette->entries[color].r;
                    context->palettes[output + 1] = palette->entries[color].g;
                    context->palettes[output + 2] = palette->entries[color].b;
                    context->counts[box * 5 + color] = cluster_counts[color];
                }
            }
        } else {
            set_error(context, 1000 + code);
        }
        if (result) api->result_destroy(result);
        if (image) api->image_destroy(image);
    }
    free(ranked);
    free(candidates);
    free(packed);
    free(indexed);
    free(rgba);
    api->attr_destroy(attribute);
    return NULL;
}

static int execute_batch(
    const char *library_path,
    batch_context *context,
    int workers
) {
    liq_api api = {0};
    int loaded = load_api(library_path, &api);
    if (loaded != 0) {
        if (api.handle) dlclose(api.handle);
        return loaded;
    }
    context->api = &api;
    atomic_init(&context->next_box, 0);
    atomic_init(&context->status, 0);
    int thread_count = workers < (int)context->box_count
        ? workers : (int)context->box_count;
    pthread_t *threads = (pthread_t *)calloc(
        thread_count > 1 ? (size_t)(thread_count - 1) : 1, sizeof(pthread_t)
    );
    if (!threads) {
        dlclose(api.handle);
        return -11;
    }
    int created = 0;
    for (int index = 0; index < thread_count - 1; index++) {
        if (pthread_create(&threads[index], NULL, batch_worker, context) == 0) {
            created++;
        } else {
            set_error(context, -12);
            break;
        }
    }
    batch_worker(context);
    for (int index = 0; index < created; index++) {
        pthread_join(threads[index], NULL);
    }
    free(threads);
    int status = atomic_load(&context->status);
    dlclose(api.handle);
    return status;
}

int liq_quantize_many(
    const char *library_path,
    const uint8_t *rgb,
    const size_t *offsets,
    size_t box_count,
    int speed,
    int workers,
    uint8_t *palettes,
    uint64_t *counts,
    uint8_t *sizes
) {
    if (!library_path || !rgb || !offsets || !palettes || !counts || !sizes ||
        box_count == 0 || speed < 1 || speed > 10 || workers < 1) return -10;
    batch_context context = {
        .rgb = rgb,
        .offsets = offsets,
        .box_count = box_count,
        .speed = speed,
        .observed = 0,
        .palettes = palettes,
        .counts = counts,
        .sizes = sizes,
    };
    return execute_batch(library_path, &context, workers);
}

int liq_quantize_many_observed(
    const char *library_path,
    const uint8_t *rgb,
    const size_t *offsets,
    size_t box_count,
    int speed,
    int workers,
    double min_cluster_ratio,
    uint8_t *palettes,
    double *weights,
    uint8_t *sizes
) {
    if (!library_path || !rgb || !offsets || !palettes || !weights || !sizes ||
        box_count == 0 || speed < 1 || speed > 10 || workers < 1 ||
        min_cluster_ratio < 0.0 || min_cluster_ratio > 1.0) return -10;
    batch_context context = {
        .rgb = rgb,
        .offsets = offsets,
        .box_count = box_count,
        .speed = speed,
        .observed = 1,
        .min_cluster_ratio = min_cluster_ratio,
        .palettes = palettes,
        .weights = weights,
        .sizes = sizes,
    };
    return execute_batch(library_path, &context, workers);
}
