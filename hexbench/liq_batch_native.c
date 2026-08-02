#include <dlfcn.h>
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
    const liq_api *api;
    const uint8_t *rgb;
    const size_t *offsets;
    size_t box_count;
    int speed;
    _Atomic size_t next_box;
    _Atomic int status;
    uint8_t *palettes;
    uint64_t *counts;
    uint8_t *sizes;
} batch_context;

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
    size_t capacity = 0;

    for (;;) {
        size_t box = atomic_fetch_add(&context->next_box, 1);
        if (box >= context->box_count) break;
        size_t start = context->offsets[box];
        size_t pixel_count = context->offsets[box + 1] - start;
        if (pixel_count > capacity) {
            free(rgba);
            free(indexed);
            rgba = (uint8_t *)malloc(pixel_count * 4);
            indexed = (uint8_t *)malloc(pixel_count);
            if (!rgba || !indexed) {
                free(rgba);
                free(indexed);
                rgba = NULL;
                indexed = NULL;
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
            context->sizes[box] = (uint8_t)palette_size;
            for (unsigned int color = 0; color < palette_size; color++) {
                size_t output = (box * 5 + color) * 3;
                context->palettes[output] = palette->entries[color].r;
                context->palettes[output + 1] = palette->entries[color].g;
                context->palettes[output + 2] = palette->entries[color].b;
            }
            for (size_t index = 0; index < pixel_count; index++) {
                uint8_t color = indexed[index];
                if (color < palette_size) {
                    context->counts[box * 5 + color]++;
                }
            }
        } else {
            set_error(context, 1000 + code);
        }
        if (result) api->result_destroy(result);
        if (image) api->image_destroy(image);
    }
    free(indexed);
    free(rgba);
    api->attr_destroy(attribute);
    return NULL;
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
    liq_api api = {0};
    int loaded = load_api(library_path, &api);
    if (loaded != 0) {
        if (api.handle) dlclose(api.handle);
        return loaded;
    }
    batch_context context = {
        .api = &api,
        .rgb = rgb,
        .offsets = offsets,
        .box_count = box_count,
        .speed = speed,
        .next_box = 0,
        .status = 0,
        .palettes = palettes,
        .counts = counts,
        .sizes = sizes,
    };
    int thread_count = workers < (int)box_count ? workers : (int)box_count;
    pthread_t *threads = (pthread_t *)calloc(
        thread_count > 1 ? (size_t)(thread_count - 1) : 1, sizeof(pthread_t)
    );
    if (!threads) {
        dlclose(api.handle);
        return -11;
    }
    int created = 0;
    for (int index = 0; index < thread_count - 1; index++) {
        if (pthread_create(&threads[index], NULL, batch_worker, &context) == 0) {
            created++;
        } else {
            set_error(&context, -12);
            break;
        }
    }
    batch_worker(&context);
    for (int index = 0; index < created; index++) {
        pthread_join(threads[index], NULL);
    }
    free(threads);
    int status = atomic_load(&context.status);
    dlclose(api.handle);
    return status;
}
