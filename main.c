/*
 * ImageInfo (imginf) — utility for extracting metadata from image files
 * Usage: imginf -<path> [arg]
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <libexif/exif-data.h>
#include <libexif/exif-tag.h>
#include <libexif/exif-entry.h>

#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_CYAN    "\033[36m"
#define C_YELLOW  "\033[33m"
#define C_GREEN   "\033[32m"
#define C_RED     "\033[31m"
#define C_BLUE    "\033[34m"
#define C_MAGENTA "\033[35m"
#define C_GRAY    "\033[90m"

/* ─── Version ─────────────────────────────────────────────────── */
#define IMGINF_VERSION "1.1.5"

/* ─── Output mode flags ────────────────────────────────────────── */
typedef struct {
    int show_basic;      /* -b  basic fields only                  */
    int show_all;        /* -a  all EXIF tags                      */
    int show_gps;        /* -g  GPS only                           */
    int show_camera;     /* -c  camera data only                   */
    int show_json;       /* -j  output in JSON format              */
    int show_size;       /* -s  file size                          */
    int detect_ai;       /* -d / --detectAi  AI generation info    */
    int no_color;        /* --no-color  without ANSI colors        */
    int verbose;         /* -v  verbose mode                       */
} Options;

/* ═══════════════════════════════════════════════════════════════
   AI DETECTION — PNG chunk reader + JPEG comment/UserComment
   ═══════════════════════════════════════════════════════════════ */

/* PNG signature */
static const unsigned char PNG_SIG[8] = {137,80,78,71,13,10,26,10};

/* Read big-endian uint32 */
static unsigned int read_be32(const unsigned char *p)
{
    return ((unsigned int)p[0] << 24) | ((unsigned int)p[1] << 16)
         | ((unsigned int)p[2] <<  8) |  (unsigned int)p[3];
}

/* ── AI result structure ── */
#define AI_STR_MAX 8192

typedef struct {
    char tool[128];          /* Detected tool name                 */
    char prompt[AI_STR_MAX]; /* Positive prompt                    */
    char neg_prompt[AI_STR_MAX]; /* Negative prompt               */
    char model[256];         /* Checkpoint / model name            */
    char loras[1024];        /* LoRA list                          */
    char seed[64];
    char steps[32];
    char cfg[32];            /* CFG scale                          */
    char sampler[128];
    char scheduler[64];
    char width[16];
    char height[16];
    char clip_skip[16];
    char vae[128];
    char denoising[32];
    char hires[256];         /* Hires fix info                     */
    char workflow[AI_STR_MAX]; /* Raw ComfyUI / InvokeAI JSON      */
    char raw_params[AI_STR_MAX]; /* Full raw parameter string      */
    int  found;              /* 1 if any AI data was found         */
} AiInfo;

/* ── Helper: safe string copy, trim whitespace ── */
static void safe_copy(char *dst, size_t dstsz, const char *src, size_t srclen)
{
    size_t n = srclen < dstsz - 1 ? srclen : dstsz - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
    /* trim trailing whitespace */
    for (int i = (int)n - 1; i >= 0 && (dst[i] == '\n' || dst[i] == '\r'
                                        || dst[i] == ' ' || dst[i] == '\t'); i--)
        dst[i] = '\0';
}


static void parse_a1111(AiInfo *ai, const char *text, size_t tlen)
{
    /* Copy raw */
    safe_copy(ai->raw_params, sizeof(ai->raw_params), text, tlen);

    /* Find "Negative prompt:" line */
    const char *neg_marker = "Negative prompt:";
    const char *neg_pos = strstr(text, neg_marker);

    /* Everything before neg_pos is positive prompt */
    size_t pos_len = neg_pos ? (size_t)(neg_pos - text) : 0;

    /* Find "Steps:" to bound positive prompt if no negative */
    const char *steps_pos = strstr(text, "\nSteps:");
    if (!neg_pos && steps_pos)
        pos_len = (size_t)(steps_pos - text);
    else if (!neg_pos)
        pos_len = tlen;

    if (pos_len > 0)
        safe_copy(ai->prompt, sizeof(ai->prompt), text, pos_len);

    /* Extract negative prompt */
    if (neg_pos) {
        neg_pos += strlen(neg_marker);
        while (*neg_pos == ' ') neg_pos++;
        const char *neg_end = strstr(neg_pos, "\nSteps:");
        if (!neg_end) neg_end = text + tlen;
        safe_copy(ai->neg_prompt, sizeof(ai->neg_prompt),
                  neg_pos, (size_t)(neg_end - neg_pos));
    }

    /* ── Key-value pairs after prompts ── */
    /* Helper macro: find "Key: value," in the params line */
    const char *params_line = steps_pos ? steps_pos + 1 : NULL;
    if (!params_line) return;

#define EXTRACT_KV(key, dst, dstsz) \
    do { \
        const char *_p = strstr(params_line, key ": "); \
        if (_p) { \
            _p += strlen(key) + 2; \
            const char *_e = strpbrk(_p, ",\n"); \
            size_t _l = _e ? (size_t)(_e - _p) : strlen(_p); \
            safe_copy(dst, dstsz, _p, _l); \
        } \
    } while(0)

    EXTRACT_KV("Steps",          ai->steps,      sizeof(ai->steps));
    EXTRACT_KV("Sampler",        ai->sampler,    sizeof(ai->sampler));
    EXTRACT_KV("Schedule type",  ai->scheduler,  sizeof(ai->scheduler));
    EXTRACT_KV("CFG scale",      ai->cfg,        sizeof(ai->cfg));
    EXTRACT_KV("Seed",           ai->seed,       sizeof(ai->seed));
    EXTRACT_KV("Model",          ai->model,      sizeof(ai->model));
    /* Append model hash in parentheses */
    {
        char mhash[64] = "";
        EXTRACT_KV("Model hash", mhash, sizeof(mhash));
        if (mhash[0] && ai->model[0]) {
            strncat(ai->model, " (", sizeof(ai->model)-strlen(ai->model)-1);
            strncat(ai->model, mhash, sizeof(ai->model)-strlen(ai->model)-1);
            strncat(ai->model, ")", sizeof(ai->model)-strlen(ai->model)-1);
        }
    }
    EXTRACT_KV("VAE",            ai->vae,        sizeof(ai->vae));
    EXTRACT_KV("Clip skip",      ai->clip_skip,  sizeof(ai->clip_skip));
    EXTRACT_KV("Denoising strength", ai->denoising, sizeof(ai->denoising));

    /* Size: WxH */
    const char *size_p = strstr(params_line, "Size: ");
    if (size_p) {
        size_p += 6;
        char tmp[32]; size_t sz = 0;
        while (size_p[sz] && size_p[sz] != ',' && size_p[sz] != '\n') sz++;
        safe_copy(tmp, sizeof(tmp), size_p, sz);
        char *x = strchr(tmp, 'x');
        if (x) { *x = '\0'; safe_copy(ai->width, 8, tmp, strlen(tmp));
                             safe_copy(ai->height, 8, x+1, strlen(x+1)); }
    }

    /* Hires fix */
    const char *hr = strstr(params_line, "Hires upscaler:");
    if (hr) {
        size_t hl = 0;
        while (hr[hl] && hr[hl] != '\n') hl++;
        safe_copy(ai->hires, sizeof(ai->hires), hr, hl);
    }

    /* LoRA: <lora:name:weight> patterns in prompt */
    const char *lp = ai->prompt;
    int lora_count = 0;
    char lora_buf[1024] = "";
    while ((lp = strstr(lp, "<lora:")) != NULL) {
        const char *le = strchr(lp, '>');
        if (!le) break;
        char lname[128];
        safe_copy(lname, sizeof(lname), lp + 1, (size_t)(le - lp - 1));
        if (lora_count > 0) strncat(lora_buf, ", ", sizeof(lora_buf)-strlen(lora_buf)-1);
        strncat(lora_buf, lname, sizeof(lora_buf)-strlen(lora_buf)-1);
        lora_count++;
        lp = le + 1;
    }
    /* Also collect Lora hashes from params line */
    const char *lh = strstr(params_line, "Lora hashes:");
    if (lh) {
        lh += 12;
        while (*lh == ' ') lh++;
        /* find end: next key-value pair (pattern: ", Word:") or newline */
        const char *lhe = lh;
        while (*lhe && *lhe != '\n') {
            /* stop before next top-level key like ", Version:" */
            if (lhe[0] == ',' && lhe[1] == ' ' && lhe[2] >= 'A' && lhe[2] <= 'Z') {
                const char *colon = strchr(lhe + 2, ':');
                const char *space = strchr(lhe + 2, ' ');
                if (colon && (!space || colon < space)) break;
            }
            lhe++;
        }
        size_t lhl = (size_t)(lhe - lh);
        if (lhl > 0) {
            if (lora_buf[0]) strncat(lora_buf, " | hashes: ", sizeof(lora_buf)-strlen(lora_buf)-1);
            strncat(lora_buf, lh, lhl < sizeof(lora_buf)-strlen(lora_buf)-1
                                  ? lhl : sizeof(lora_buf)-strlen(lora_buf)-1);
        }
    }
    if (lora_buf[0]) safe_copy(ai->loras, sizeof(ai->loras), lora_buf, strlen(lora_buf));

#undef EXTRACT_KV
}

/* ── Detect tool from raw text ── */
static void detect_tool(AiInfo *ai, const char *key, const char *val, size_t vlen)
{
    (void)vlen;
    /* A1111 / Forge / SD.Next */
    if (strcmp(key, "parameters") == 0) {
        if (!ai->tool[0]) strcpy(ai->tool, "Stable Diffusion (A1111/WebUI/Forge)");
        parse_a1111(ai, val, strlen(val));
        ai->found = 1;
        return;
    }
    /* NovelAI */
    if (strcmp(key, "Comment") == 0 && strstr(val, "\"source\"")) {
        if (!ai->tool[0]) strcpy(ai->tool, "NovelAI");
        safe_copy(ai->workflow, sizeof(ai->workflow), val, strlen(val));
        ai->found = 1;
        return;
    }
    /* InvokeAI */
    if (strcmp(key, "invokeai_metadata") == 0 || strcmp(key, "invokeai") == 0) {
        if (!ai->tool[0]) strcpy(ai->tool, "InvokeAI");
        safe_copy(ai->workflow, sizeof(ai->workflow), val, strlen(val));
        ai->found = 1;
        return;
    }
    /* ComfyUI — stores full node graph as JSON in "prompt" and "workflow" chunks */
    if (strcmp(key, "prompt") == 0 && (strstr(val, "\"class_type\"")
                                    || strstr(val, "KSampler"))) {
        if (!ai->tool[0]) strcpy(ai->tool, "ComfyUI");
        safe_copy(ai->workflow, sizeof(ai->workflow), val, strlen(val));
        ai->found = 1;
        /* Try to extract sampler/steps/cfg from ComfyUI JSON */
        const char *sp = strstr(val, "\"steps\":");
        if (sp) { sp += 8; size_t i=0;
                  while(sp[i] && sp[i]!=',' && sp[i]!='}') i++;
                  safe_copy(ai->steps, sizeof(ai->steps), sp, i); }
        sp = strstr(val, "\"cfg\":");
        if (sp) { sp += 6; size_t i=0;
                  while(sp[i] && sp[i]!=',' && sp[i]!='}') i++;
                  safe_copy(ai->cfg, sizeof(ai->cfg), sp, i); }
        sp = strstr(val, "\"seed\":");
        if (sp) { sp += 7; size_t i=0;
                  while(sp[i] && sp[i]!=',' && sp[i]!='}') i++;
                  safe_copy(ai->seed, sizeof(ai->seed), sp, i); }
        sp = strstr(val, "\"sampler_name\":");
        if (sp) { sp += 16; if(*sp=='"') sp++;
                  size_t i=0; while(sp[i] && sp[i]!='"') i++;
                  safe_copy(ai->sampler, sizeof(ai->sampler), sp, i); }
        return;
    }
    if (strcmp(key, "workflow") == 0 && strstr(val, "\"nodes\"")) {
        if (!ai->tool[0]) strcpy(ai->tool, "ComfyUI");
        if (!ai->workflow[0])
            safe_copy(ai->workflow, sizeof(ai->workflow), val, strlen(val));
        ai->found = 1;
        return;
    }
    /* DALL-E / Midjourney usually leave Software tag */
    if (strcmp(key, "Software") == 0) {
        if (strstr(val, "DALL-E") || strstr(val, "dalle")) {
            if (!ai->tool[0]) { snprintf(ai->tool, sizeof(ai->tool),
                                         "DALL-E (%s)", val); ai->found = 1; }
        }
        if (strstr(val, "Midjourney")) {
            if (!ai->tool[0]) { snprintf(ai->tool, sizeof(ai->tool),
                                         "Midjourney (%s)", val); ai->found = 1; }
        }
    }
    /* Generic "AI generated" XMP marker */
    if (strcmp(key, "XML:com.adobe.xmp") == 0 && strstr(val, "GenerativeAI")) {
        if (!ai->tool[0]) strcpy(ai->tool, "Adobe Firefly / Generative AI");
        ai->found = 1;
    }
}

/* ── Read PNG tEXt / iTXt chunks ── */
static void parse_png_chunks(AiInfo *ai, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return;

    unsigned char sig[8];
    if (fread(sig, 1, 8, f) != 8 || memcmp(sig, PNG_SIG, 8) != 0) {
        fclose(f); return;
    }

    unsigned char hdr[8];
    while (fread(hdr, 1, 8, f) == 8) {
        unsigned int clen = read_be32(hdr);       /* chunk data length */
        char ctype[5];
        memcpy(ctype, hdr + 4, 4); ctype[4] = '\0';

        /* Only care about tEXt and iTXt */
        if ((strcmp(ctype, "tEXt") == 0 || strcmp(ctype, "iTXt") == 0)
             && clen > 0 && clen < 1024*1024*4)
        {
            char *data = (char *)malloc(clen + 1);
            if (!data) { fseek(f, (long)(clen + 4), SEEK_CUR); continue; }
            if (fread(data, 1, clen, f) != clen) { free(data); break; }
            data[clen] = '\0';

            /* tEXt: key\0value   iTXt: key\0flags\0lang\0...\0value */
            char *key = data;
            char *val = NULL;
            size_t klen = strlen(key);

            if (strcmp(ctype, "tEXt") == 0 && klen < clen) {
                val = data + klen + 1;
            } else if (strcmp(ctype, "iTXt") == 0 && klen < clen) {
                /* skip: compression_flag(1) + compression_method(1) + lang_tag\0 + translated_key\0 */
                char *p = data + klen + 1; /* past key\0 */
                if (p < data + clen) p++;  /* compression flag */
                if (p < data + clen) p++;  /* compression method */
                { while (p < data + clen && *p) p++; if (p < data + clen) p++; } /* lang\0 */
                { while (p < data + clen && *p) p++; if (p < data + clen) p++; } /* translated key\0 */
                if (p < data + clen) val = p;
            }

            if (val && val[0])
                detect_tool(ai, key, val, (size_t)(data + clen - val));

            /* CRC */
            fseek(f, 4, SEEK_CUR);
            free(data);
        } else {
            /* Skip chunk data + CRC */
            fseek(f, (long)(clen + 4), SEEK_CUR);
        }

        if (strcmp(ctype, "IEND") == 0) break;
    }
    fclose(f);
}

/* ── Read JPEG comments and EXIF UserComment / ImageDescription ── */
static void parse_jpeg_ai(AiInfo *ai, const char *path)
{
    /* Check via libexif for common AI tags */
    ExifData *ed = exif_data_new_from_file(path);
    if (!ed) return;

    char buf[AI_STR_MAX];

    /* UserComment */
    ExifEntry *uc = exif_content_get_entry(ed->ifd[EXIF_IFD_EXIF],
                                           EXIF_TAG_USER_COMMENT);
    if (uc) {
        exif_entry_get_value(uc, buf, sizeof(buf));
        /* Only proceed if it contains actual AI parameter keywords */
        if (buf[0] && (strstr(buf, "Steps:") || strstr(buf, "Seed:")
                    || strstr(buf, "parameters") || strstr(buf, "Model:")))
            detect_tool(ai, "parameters", buf, strlen(buf));
    }

    /* ImageDescription */
    ExifEntry *id = exif_content_get_entry(ed->ifd[EXIF_IFD_0],
                                           EXIF_TAG_IMAGE_DESCRIPTION);
    if (id) {
        exif_entry_get_value(id, buf, sizeof(buf));
        if (buf[0] && strlen(buf) > 20) {
            if (strstr(buf, "Steps:") || strstr(buf, "Seed:"))
                detect_tool(ai, "parameters", buf, strlen(buf));
            else if (strstr(buf, "DALL-E") || strstr(buf, "Midjourney")
                  || strstr(buf, "Firefly") || strstr(buf, "AI"))
                detect_tool(ai, "Software", buf, strlen(buf));
        }
    }

    /* Software tag */
    ExifEntry *sw = exif_content_get_entry(ed->ifd[EXIF_IFD_0], EXIF_TAG_SOFTWARE);
    if (sw) {
        exif_entry_get_value(sw, buf, sizeof(buf));
        detect_tool(ai, "Software", buf, strlen(buf));
    }

    exif_data_unref(ed);

    /* Also scan raw JPEG COM markers */
    FILE *f = fopen(path, "rb");
    if (!f) return;
    unsigned char marker[2];
    /* Skip SOI */
    if (fread(marker, 1, 2, f) != 2) { fclose(f); return; }
    while (fread(marker, 1, 2, f) == 2) {
        if (marker[0] != 0xFF) break;
        unsigned char lenbuf[2];
        if (fread(lenbuf, 1, 2, f) != 2) break;
        unsigned int seglen = ((unsigned int)lenbuf[0] << 8) | lenbuf[1];
        if (seglen < 2) break;
        unsigned int datalen = seglen - 2;
        if (marker[1] == 0xFE && datalen > 0 && datalen < 1024*512) {
            /* COM marker */
            char *com = (char *)malloc(datalen + 1);
            if (com && fread(com, 1, datalen, f) == datalen) {
                com[datalen] = '\0';
                if (strstr(com, "Steps:") || strstr(com, "Seed:"))
                    detect_tool(ai, "parameters", com, datalen);
            }
            free(com);
        } else {
            fseek(f, datalen, SEEK_CUR);
        }
        if (marker[1] == 0xDA) break; /* SOS — image data starts */
    }
    fclose(f);
}

/* ── Print a long string with word-wrap at ~80 cols ── */
static void print_wrapped(const char *text, int indent, const Options *o)
{
    if (!text || !text[0]) return;
    const int maxcol = 80;
    int col = indent;
    const char *p = text;
    /* print initial indent */
    for (int i = 0; i < indent; i++) putchar(' ');
    while (*p) {
        if (*p == '\n') {
            putchar('\n');
            for (int i = 0; i < indent; i++) putchar(' ');
            col = indent; p++;
            continue;
        }
        /* find next word */
        const char *word_end = p;
        while (*word_end && *word_end != ' ' && *word_end != '\n') word_end++;
        int wlen = (int)(word_end - p);
        if (col + wlen + 1 > maxcol && col > indent) {
            putchar('\n');
            for (int i = 0; i < indent; i++) putchar(' ');
            col = indent;
        }
        while (p < word_end) { putchar(*p++); col++; }
        if (*p == ' ') { putchar(' '); col++; p++; }
    }
    putchar('\n');
    (void)o;
}

/* ── Print AI detection results ── */
static void print_ai_info(const AiInfo *ai, const Options *o)
{
    if (!o->show_json) {
        if (o->no_color)
            printf("\n--- AI Generation Info ---\n");
        else
            printf("\n" C_MAGENTA C_BOLD "▶ AI Generation Info" C_RESET "\n");
    }

    if (!ai->found) {
        if (o->show_json)
            printf(",\n  \"ai_detection\": {\"found\": false}");
        else {
            if (o->no_color)
                printf("  [AI] No AI generation metadata found.\n");
            else
                printf("  " C_GRAY "[AI] No AI generation metadata found in this file.\n"
                       "       (Metadata may have been stripped or the file is a real photo)\n"
                       C_RESET);
        }
        return;
    }

#define AI_FIELD(label, val) \
    do { if ((val)[0]) { \
        if (o->no_color) printf("  %-22s %s\n", label":", val); \
        else printf("  " C_CYAN "%-22s" C_RESET " %s\n", label":", val); \
    }} while(0)

    if (o->show_json) {
        printf(",\n  \"ai_detection\": {\n");
        printf("    \"found\": true");
        if (ai->tool[0])       printf(",\n    \"tool\": \"%s\"", ai->tool);
        if (ai->model[0])      printf(",\n    \"model\": \"%s\"", ai->model);
        if (ai->seed[0])       printf(",\n    \"seed\": \"%s\"", ai->seed);
        if (ai->steps[0])      printf(",\n    \"steps\": \"%s\"", ai->steps);
        if (ai->cfg[0])        printf(",\n    \"cfg_scale\": \"%s\"", ai->cfg);
        if (ai->sampler[0])    printf(",\n    \"sampler\": \"%s\"", ai->sampler);
        if (ai->scheduler[0])  printf(",\n    \"scheduler\": \"%s\"", ai->scheduler);
        if (ai->width[0])      printf(",\n    \"width\": \"%s\"", ai->width);
        if (ai->height[0])     printf(",\n    \"height\": \"%s\"", ai->height);
        if (ai->vae[0])        printf(",\n    \"vae\": \"%s\"", ai->vae);
        if (ai->clip_skip[0])  printf(",\n    \"clip_skip\": \"%s\"", ai->clip_skip);
        if (ai->denoising[0])  printf(",\n    \"denoising_strength\": \"%s\"", ai->denoising);
        if (ai->loras[0])      printf(",\n    \"loras\": \"%s\"", ai->loras);
        /* Escape quotes in prompt for JSON */
        if (ai->prompt[0]) {
            printf(",\n    \"prompt\": \"");
            for (const char *p = ai->prompt; *p; p++) {
                if (*p == '"') printf("\\\"");
                else if (*p == '\\') printf("\\\\");
                else if (*p == '\n') printf("\\n");
                else putchar(*p);
            }
            printf("\"");
        }
        if (ai->neg_prompt[0]) {
            printf(",\n    \"negative_prompt\": \"");
            for (const char *p = ai->neg_prompt; *p; p++) {
                if (*p == '"') printf("\\\"");
                else if (*p == '\\') printf("\\\\");
                else if (*p == '\n') printf("\\n");
                else putchar(*p);
            }
            printf("\"");
        }
        printf("\n  }");
        return;
    }

    AI_FIELD("Tool / Generator", ai->tool);
    AI_FIELD("Model",            ai->model);
    AI_FIELD("Seed",             ai->seed);
    AI_FIELD("Steps",            ai->steps);
    AI_FIELD("CFG Scale",        ai->cfg);
    AI_FIELD("Sampler",          ai->sampler);
    AI_FIELD("Scheduler",        ai->scheduler);
    if (ai->width[0] && ai->height[0]) {
        char dim[48]; snprintf(dim, sizeof(dim), "%s × %s", ai->width, ai->height);
        if (o->no_color) printf("  %-22s %s\n", "Resolution:", dim);
        else printf("  " C_CYAN "%-22s" C_RESET " %s\n", "Resolution:", dim);
    }
    AI_FIELD("VAE",              ai->vae);
    AI_FIELD("Clip skip",        ai->clip_skip);
    AI_FIELD("Denoising",        ai->denoising);
    AI_FIELD("Hires fix",        ai->hires);

    /* LoRAs */
    if (ai->loras[0]) {
        if (o->no_color) printf("  %-22s %s\n", "LoRA(s):", ai->loras);
        else printf("  " C_CYAN "%-22s" C_RESET " " C_GREEN "%s" C_RESET "\n",
                    "LoRA(s):", ai->loras);
    }

    /* Prompt */
    if (ai->prompt[0]) {
        if (o->no_color) printf("  Prompt:\n");
        else printf("  " C_CYAN "Prompt:" C_RESET "\n");
        print_wrapped(ai->prompt, 4, o);
    }

    /* Negative prompt */
    if (ai->neg_prompt[0]) {
        if (o->no_color) printf("  Negative prompt:\n");
        else printf("  " C_CYAN "Negative prompt:" C_RESET "\n");
        if (!o->no_color) printf(C_RED);
        print_wrapped(ai->neg_prompt, 4, o);
        if (!o->no_color) printf(C_RESET);
    }

    /* ComfyUI / InvokeAI workflow (truncated) */
    if (ai->workflow[0]) {
        if (o->no_color)
            printf("  Workflow / Raw JSON (truncated):\n");
        else
            printf("  " C_CYAN "Workflow / Raw JSON:" C_RESET C_GRAY
                   " (truncated to 400 chars)" C_RESET "\n");
        char trunc[401];
        safe_copy(trunc, sizeof(trunc), ai->workflow, 400);
        printf("    %s...\n", trunc);
    }

#undef AI_FIELD
}

/* ─── Helper functions ────────────────────────────────── */

static void print_banner(const Options *o)
{
    if (o->show_json) return;
    if (o->no_color) {
        printf("========================================\n");
        printf("  ImageInfo v%s\n", IMGINF_VERSION);
        printf("========================================\n");
    } else {
        printf(C_CYAN C_BOLD
               "╔══════════════════════════════════════╗\n"
               "║      ImageInfo v%-6s               ║\n"
               "╚══════════════════════════════════════╝\n"
               C_RESET, IMGINF_VERSION);
    }
}

static void print_help(void)
{
    printf(C_BOLD "ImageInfo v%s" C_RESET
           " — utility for extracting metadata from images\n\n", IMGINF_VERSION);

    printf(C_YELLOW "Usage:" C_RESET "\n");
    printf("  imginf -<path_to_file> [options]\n\n");

    printf(C_YELLOW "Options:" C_RESET "\n");
    printf("  " C_GREEN "-b" C_RESET
           "            Basic fields only (extension, date, size)\n");
    printf("  " C_GREEN "-a" C_RESET
           "            All available EXIF tags\n");
    printf("  " C_GREEN "-g" C_RESET
           "            GPS / location only\n");
    printf("  " C_GREEN "-c" C_RESET
           "            Camera data only (manufacturer, model, ISO...)\n");
    printf("  " C_GREEN "-s" C_RESET
           "            Detailed file size information\n");
    printf("  " C_GREEN "-j" C_RESET
           "            Output in JSON format\n");
    printf("  " C_GREEN "-d" C_RESET
           " / " C_GREEN "--detectAi" C_RESET
           "  Detect AI generation metadata:\n");
    printf("               tool, prompt, negative prompt, LoRA(s),\n");
    printf("               model, seed, CFG, sampler, steps, VAE,\n");
    printf("               clip skip, denoising, hires fix\n");
    printf("               Supports: A1111, Forge, ComfyUI, NovelAI,\n");
    printf("                         InvokeAI, DALL-E, Midjourney\n");
    printf("  " C_GREEN "-v" C_RESET
           "            Verbose mode\n");
    printf("  " C_GREEN "--no-color" C_RESET
           "    Disable colored output\n");
    printf("  " C_GREEN "--help" C_RESET
           "        Show this help message\n");
    printf("  " C_GREEN "--version" C_RESET
           "     Program version\n\n");

    printf(C_YELLOW "Examples:" C_RESET "\n");
    printf("  imginf -/home/user/photo.jpg\n");
    printf("  imginf -./img.png -g\n");
    printf("  imginf -/tmp/shot.jpeg -j\n");
    printf("  imginf -photo.jpg -c -v\n");
    printf("  imginf -photo.jpg -a --no-color\n");
    printf("  imginf -ai_image.png -d\n");
    printf("  imginf -ai_image.png -d -j\n\n");

    printf(C_YELLOW "Supported formats:" C_RESET
           " JPEG, PNG, TIFF, HEIC, WEBP\n");
}

/* Get file extension (including the dot) */
static const char *get_extension(const char *path)
{
    const char *dot = strrchr(path, '.');
    if (!dot || dot == path) return "(no extension)";
    return dot;
}

/* Format bytes into a human-readable form */
static void format_size(long bytes, char *buf, size_t bufsz)
{
    if (bytes < 1024)
        snprintf(buf, bufsz, "%ld B", bytes);
    else if (bytes < 1024 * 1024)
        snprintf(buf, bufsz, "%.2f KB", bytes / 1024.0);
    else if (bytes < 1024 * 1024 * 1024)
        snprintf(buf, bufsz, "%.2f MB", bytes / (1024.0 * 1024));
    else
        snprintf(buf, bufsz, "%.2f GB", bytes / (1024.0 * 1024 * 1024));
}

/* Convert EXIF GPS rational value to decimal degrees */
static double rational_to_double(ExifEntry *e, int idx)
{
    ExifRational r;
    r = exif_get_rational(e->data + idx * 8,
                          exif_data_get_byte_order(e->parent->parent));
    if (r.denominator == 0) return 0.0;
    return (double)r.numerator / (double)r.denominator;
}

/* GPS DMS → decimal degrees */
static double gps_dms_to_decimal(ExifEntry *e)
{
    double deg = rational_to_double(e, 0);
    double min = rational_to_double(e, 1);
    double sec = rational_to_double(e, 2);
    return deg + min / 60.0 + sec / 3600.0;
}

/* Approximate country by coordinates (simplified table) */
static const char *approx_country(double lat, double lon)
{
    if (lat >= 41.0 && lat <= 71.5 && lon >= 19.0 && lon <= 180.0)
        return "Russia / CIS";
    if (lat >= 44.0 && lat <= 52.5 && lon >= 22.0 && lon <= 40.0)
        return "Ukraine";
    if (lat >= 51.0 && lat <= 71.0 && lon >= -8.0 && lon <= 2.0)
        return "United Kingdom / Ireland";
    if (lat >= 41.0 && lat <= 51.0 && lon >= -5.0 && lon <= 9.5)
        return "France / Spain";
    if (lat >= 47.0 && lat <= 55.0 && lon >= 6.0  && lon <= 15.0)
        return "Germany / Austria";
    if (lat >= 24.0 && lat <= 50.0 && lon >= -125.0 && lon <= -66.0)
        return "USA";
    if (lat >= 42.0 && lat <= 83.0 && lon >= -141.0 && lon <= -52.0)
        return "Canada";
    if (lat >= -55.0 && lat <= -15.0 && lon >= -74.0 && lon <= -35.0)
        return "Brazil";
    if (lat >= 18.0 && lat <= 53.0 && lon >= 73.0  && lon <= 135.0)
        return "China / Asia";
    if (lat >= 30.0 && lat <= 37.0 && lon >= 129.0 && lon <= 146.0)
        return "Japan";
    if (lat >= -43.0 && lat <= -11.0 && lon >= 113.0 && lon <= 154.0)
        return "Australia";
    return "Unknown";
}

/* ─── Field output ─────────────────────────────────────────────── */

#define FIELD(label, fmt, ...) \
    do { \
        if (o->no_color) \
            printf("%-22s " fmt "\n", label":", ##__VA_ARGS__); \
        else \
            printf(C_CYAN "%-22s" C_RESET " " fmt "\n", label":", ##__VA_ARGS__); \
    } while(0)

#define SECTION(title) \
    do { \
        if (!o->show_json) { \
            if (o->no_color) \
                printf("\n--- %s ---\n", title); \
            else \
                printf("\n" C_YELLOW C_BOLD "▶ %s" C_RESET "\n", title); \
        } \
    } while(0)

/* ─── Extract tag as string ─────────────────────────────── */
static int exif_tag_str(ExifData *ed, ExifIfd ifd, ExifTag tag,
                         char *buf, size_t bufsz)
{
    ExifEntry *e = exif_content_get_entry(ed->ifd[ifd], tag);
    if (!e) return 0;
    exif_entry_get_value(e, buf, (unsigned int)bufsz);
    return 1;
}

/* ─── Main file processing ───────────────────────────────── */
static void process_file(const char *path, const Options *o)
{
    /* Existence check */
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, C_RED "Error:" C_RESET " file not found: %s\n", path);
        return;
    }
    if (!S_ISREG(st.st_mode)) {
        fprintf(stderr, C_RED "Error:" C_RESET " is not a file: %s\n", path);
        return;
    }

    print_banner(o);

    const char *ext = get_extension(path);

    char timebuf[64];
    struct tm *tm_info = localtime(&st.st_mtime);
    strftime(timebuf, sizeof(timebuf), "%d.%m.%Y %H:%M:%S", tm_info);

    char sizebuf[32];
    format_size((long)st.st_size, sizebuf, sizeof(sizebuf));

    /* ── JSON mode header ── */
    if (o->show_json) {
        printf("{\n");
        printf("  \"file\": \"%s\",\n", path);
        printf("  \"extension\": \"%s\",\n", ext);
        printf("  \"size_bytes\": %ld,\n", (long)st.st_size);
        printf("  \"size_human\": \"%s\",\n", sizebuf);
        printf("  \"mtime\": \"%s\"", timebuf);
    }

    /* ── Regular output ── */
    if (!o->show_json) {
        if (!o->show_gps && !o->show_camera && !o->detect_ai) {
            SECTION("General Information");
            FIELD("Extension",    "%s", ext);
            FIELD("File path",    "%s", path);
            FIELD("File size",    "%s (%ld bytes)", sizebuf, (long)st.st_size);
            FIELD("Last modified","%s", timebuf);
        }
    }

    /* ── AI Detection (-d) ── */
    if (o->detect_ai) {
        AiInfo ai;
        memset(&ai, 0, sizeof(ai));

        /* Dispatch by extension */
        const char *lext = ext;
        char extlow[16] = "";
        for (int i = 0; lext[i] && i < 14; i++)
            extlow[i] = (char)tolower((unsigned char)lext[i]);

        if (strcmp(extlow, ".png") == 0)
            parse_png_chunks(&ai, path);
        else
            parse_jpeg_ai(&ai, path);   /* JPEG / TIFF / WEBP */

        /* For PNG also check EXIF (some tools embed both) */
        if (strcmp(extlow, ".png") == 0)
            parse_jpeg_ai(&ai, path);

        print_ai_info(&ai, o);

        /* If only -d flag, close JSON and return */
        if (!o->show_all && !o->show_gps && !o->show_camera && !o->show_basic) {
            if (o->show_json) printf("\n}\n");
            else {
                if (!o->no_color)
                    printf("\n" C_GRAY "────────────────────────────────────────\n" C_RESET);
                else
                    printf("\n----------------------------------------\n");
            }
            return;
        }
    }

    /* ── Load EXIF ── */
    ExifData *ed = exif_data_new_from_file(path);
    if (!ed) {
        if (!o->show_json) {
            if (o->no_color)
                printf("\n[!] EXIF data not found in file.\n");
            else
                printf("\n" C_RED "[!]" C_RESET
                       " EXIF data not found in this file.\n"
                       C_GRAY "    Metadata may have been removed.\n" C_RESET);
        } else {
            printf("\n}\n");
        }
        return;
    }

    char buf[256];

    /* ════════════════════════════════════════
       SECTION: CAMERA
       ════════════════════════════════════════ */
    if (!o->show_gps && (o->show_camera || o->show_all || 1)) {
        int has_camera = 0;
        char make[128]="", model[128]="", lens[128]="",
             software[128]="", orientation[64]="";
        char iso[32]="", exp_time[64]="", aperture[64]="",
             focal[64]="", focal35[64]="", flash[64]="",
             wb[64]="", metering[64]="", exp_prog[64]="";

        has_camera |= exif_tag_str(ed, EXIF_IFD_0, EXIF_TAG_MAKE, make, sizeof(make));
        has_camera |= exif_tag_str(ed, EXIF_IFD_0, EXIF_TAG_MODEL, model, sizeof(model));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_LENS_MODEL, lens, sizeof(lens));
        exif_tag_str(ed, EXIF_IFD_0, EXIF_TAG_SOFTWARE, software, sizeof(software));
        exif_tag_str(ed, EXIF_IFD_0, EXIF_TAG_ORIENTATION, orientation, sizeof(orientation));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_ISO_SPEED_RATINGS, iso, sizeof(iso));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_TIME, exp_time, sizeof(exp_time));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_FNUMBER, aperture, sizeof(aperture));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_FOCAL_LENGTH, focal, sizeof(focal));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_FOCAL_LENGTH_IN_35MM_FILM, focal35, sizeof(focal35));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_FLASH, flash, sizeof(flash));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_WHITE_BALANCE, wb, sizeof(wb));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_METERING_MODE, metering, sizeof(metering));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_EXPOSURE_PROGRAM, exp_prog, sizeof(exp_prog));

        if (has_camera && !o->show_json) {
            if (!o->show_basic) {
                SECTION("Camera Information");
                if (make[0])        FIELD("Make",          "%s", make);
                if (model[0])       FIELD("Model",         "%s", model);
                if (lens[0])        FIELD("Lens",          "%s", lens);
                if (software[0])    FIELD("Software",      "%s", software);
                if (orientation[0]) FIELD("Orientation",   "%s", orientation);
                if (iso[0])         FIELD("ISO",           "%s", iso);
                if (exp_time[0])    FIELD("Exposure time", "%s s", exp_time);
                if (aperture[0])    FIELD("Aperture",      "f/%s", aperture);
                if (focal[0])       FIELD("Focal length",  "%s mm", focal);
                if (focal35[0])     FIELD("Focal (35mm)",  "%s mm", focal35);
                if (flash[0])       FIELD("Flash",         "%s", flash);
                if (wb[0])          FIELD("White balance", "%s", wb);
                if (metering[0])    FIELD("Metering mode", "%s", metering);
                if (exp_prog[0])    FIELD("Exposure prog", "%s", exp_prog);
            }
        }

        if (o->show_json && has_camera) {
            printf(",\n  \"camera\": {\n");
            int first = 1;
#define JP(k,v) if ((v)[0]) { printf(first?"":",\n"); printf("    \"%s\": \"%s\"",k,v); first=0; }
            JP("make", make) JP("model", model) JP("lens", lens)
            JP("software", software) JP("iso", iso) JP("exposure_time", exp_time)
            JP("aperture", aperture) JP("focal_length", focal) JP("flash", flash)
#undef JP
            printf("\n  }");
        }
    }

    /* ════════════════════════════════════════
       SECTION: DATE / TIME TAKEN
       ════════════════════════════════════════ */
    if (!o->show_gps && !o->show_camera) {
        char dt_orig[64]="", dt_digit[64]="";
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_DATE_TIME_ORIGINAL,  dt_orig,  sizeof(dt_orig));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_DATE_TIME_DIGITIZED, dt_digit, sizeof(dt_digit));

        if (!o->show_json) {
            if (!o->show_basic) SECTION("Date and Time Taken");
            if (dt_orig[0])  FIELD("Time created",   "%s", dt_orig);
            if (dt_digit[0]) FIELD("Time digitized", "%s", dt_digit);
        }
        if (o->show_json) {
            if (dt_orig[0])  printf(",\n  \"time_created\": \"%s\"", dt_orig);
            if (dt_digit[0]) printf(",\n  \"time_digitized\": \"%s\"", dt_digit);
        }
    }

    /* ════════════════════════════════════════
       SECTION: IMAGE RESOLUTION
       ════════════════════════════════════════ */
    if (!o->show_gps && !o->show_camera && !o->show_basic) {
        char px_x[32]="", px_y[32]="", res_x[32]="", res_y[32]="", res_unit[32]="";
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_PIXEL_X_DIMENSION, px_x, sizeof(px_x));
        exif_tag_str(ed, EXIF_IFD_EXIF, EXIF_TAG_PIXEL_Y_DIMENSION, px_y, sizeof(px_y));
        exif_tag_str(ed, EXIF_IFD_0,    EXIF_TAG_X_RESOLUTION,       res_x, sizeof(res_x));
        exif_tag_str(ed, EXIF_IFD_0,    EXIF_TAG_Y_RESOLUTION,       res_y, sizeof(res_y));
        exif_tag_str(ed, EXIF_IFD_0,    EXIF_TAG_RESOLUTION_UNIT,    res_unit, sizeof(res_unit));

        if (!o->show_json) {
            if (px_x[0] || px_y[0]) {
                SECTION("Image Resolution");
                if (px_x[0] && px_y[0]) FIELD("Dimensions", "%s × %s px", px_x, px_y);
                if (res_x[0] && res_y[0])
                    FIELD("DPI", "%s × %s (%s)", res_x, res_y,
                          res_unit[0] ? res_unit : "?");
            }
        }
        if (o->show_json && px_x[0] && px_y[0]) {
            printf(",\n  \"dimensions\": \"%s x %s\"", px_x, px_y);
            if (res_x[0]) printf(",\n  \"dpi_x\": \"%s\"", res_x);
            if (res_y[0]) printf(",\n  \"dpi_y\": \"%s\"", res_y);
        }
    }

    /* ════════════════════════════════════════
       SECTION: GPS
       ════════════════════════════════════════ */
    {
        ExifEntry *lat_e = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS],
                                                  (ExifTag)EXIF_TAG_GPS_LATITUDE);
        ExifEntry *lat_r = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS],
                                                  (ExifTag)EXIF_TAG_GPS_LATITUDE_REF);
        ExifEntry *lon_e = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS],
                                                  (ExifTag)EXIF_TAG_GPS_LONGITUDE);
        ExifEntry *lon_r = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS],
                                                  (ExifTag)EXIF_TAG_GPS_LONGITUDE_REF);
        ExifEntry *alt_e = exif_content_get_entry(ed->ifd[EXIF_IFD_GPS],
                                                  (ExifTag)EXIF_TAG_GPS_ALTITUDE);

        if (lat_e && lon_e) {
            double lat_d = gps_dms_to_decimal(lat_e);
            double lon_d = gps_dms_to_decimal(lon_e);

            if (lat_r) {
                exif_entry_get_value(lat_r, buf, sizeof(buf));
                if (buf[0] == 'S' || buf[0] == 's') lat_d = -lat_d;
            }
            if (lon_r) {
                exif_entry_get_value(lon_r, buf, sizeof(buf));
                if (buf[0] == 'W' || buf[0] == 'w') lon_d = -lon_d;
            }

            char alt_str[64] = "";
            if (alt_e) exif_entry_get_value(alt_e, alt_str, sizeof(alt_str));

            const char *country = approx_country(lat_d, lon_d);

            if (!o->show_json) {
                SECTION("GPS / Location");
                FIELD("Approximate location", "%s", country);
                FIELD("Coordinates",
                      "%.6f°%c, %.6f°%c",
                      fabs(lat_d), lat_d >= 0 ? 'N' : 'S',
                      fabs(lon_d), lon_d >= 0 ? 'E' : 'W');
                if (alt_str[0]) FIELD("Altitude", "%s m", alt_str);
                if (!o->no_color)
                    printf(C_GRAY "  → Google Maps: https://maps.google.com/?q=%.6f,%.6f\n"
                           C_RESET, lat_d, lon_d);
                else
                    printf("  Google Maps: https://maps.google.com/?q=%.6f,%.6f\n",
                           lat_d, lon_d);
            } else {
                printf(",\n  \"gps\": {\n");
                printf("    \"latitude\": %.6f,\n", lat_d);
                printf("    \"longitude\": %.6f,\n", lon_d);
                printf("    \"approx_country\": \"%s\"", country);
                if (alt_str[0]) printf(",\n    \"altitude\": \"%s\"", alt_str);
                printf("\n  }");
            }
        } else {
            if (!o->show_json && (o->show_gps || o->verbose)) {
                if (o->no_color)
                    printf("\n[GPS] Location data not found.\n");
                else
                    printf("\n" C_GRAY "[GPS] Location data unavailable.\n" C_RESET);
            }
        }
    }

    /* ════════════════════════════════════════
       SECTION: ALL EXIF TAGS (-a)
       ════════════════════════════════════════ */
    if (o->show_all && !o->show_json) {
        SECTION("All EXIF Tags");
        for (int i = 0; i < EXIF_IFD_COUNT; i++) {
            ExifContent *content = ed->ifd[i];
            if (!content || !content->count) continue;
            for (unsigned int j = 0; j < content->count; j++) {
                ExifEntry *e = content->entries[j];
                char val[512] = "";
                exif_entry_get_value(e, val, sizeof(val));
                const char *tag_name = exif_tag_get_name_in_ifd(e->tag, (ExifIfd)i);
                if (!tag_name) tag_name = "(unknown)";
                if (o->no_color)
                    printf("  %-32s %s\n", tag_name, val);
                else
                    printf("  " C_BLUE "%-32s" C_RESET " %s\n", tag_name, val);
            }
        }
    }

    /* ── Close ── */
    if (o->show_json) {
        printf("\n}\n");
    } else {
        if (!o->no_color)
            printf("\n" C_GRAY "────────────────────────────────────────\n" C_RESET);
        else
            printf("\n----------------------------------------\n");
    }

    exif_data_unref(ed);
}

/* ─── MAIN ───────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            C_RED "Error:" C_RESET " no file or flag specified.\n"
            "Use " C_GREEN "--help" C_RESET " for help.\n");
        return EXIT_FAILURE;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help();
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "--version") == 0) {
            printf("ImageInfo v%s\n", IMGINF_VERSION);
            return EXIT_SUCCESS;
        }
    }

    const char *filepath = NULL;
    Options opts = {0};

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        /* Long options */
        if (strcmp(arg, "--detectAi") == 0 || strcmp(arg, "--detect-ai") == 0) {
            opts.detect_ai = 1; continue;
        }
        if (strcmp(arg, "--no-color") == 0) {
            opts.no_color = 1; continue;
        }

        /* Path: starts with '-' but second char is not a known option letter */
        if (arg[0] == '-' && arg[1] != '-' && arg[1] != '\0') {
            const char single_opts[] = "bagcsjvd";
            int found = 0;
            for (int k = 0; single_opts[k]; k++) {
                if (arg[1] == single_opts[k] && arg[2] == '\0') {
                    found = 1; break;
                }
            }
            if (!found) {
                filepath = arg + 1;
                continue;
            }
        }

        /* Single-letter options */
        if (arg[0] == '-' && arg[1] != '-') {
            for (int k = 1; arg[k]; k++) {
                switch (arg[k]) {
                    case 'b': opts.show_basic  = 1; break;
                    case 'a': opts.show_all    = 1; break;
                    case 'g': opts.show_gps    = 1; break;
                    case 'c': opts.show_camera = 1; break;
                    case 's': opts.show_size   = 1; break;
                    case 'j': opts.show_json   = 1; break;
                    case 'd': opts.detect_ai   = 1; break;
                    case 'v': opts.verbose     = 1; break;
                    default:
                        fprintf(stderr, C_RED "Unknown option:" C_RESET " -%c\n", arg[k]);
                        return EXIT_FAILURE;
                }
            }
        }
    }

    if (!filepath) {
        fprintf(stderr,
            C_RED "Error:" C_RESET " file path not specified.\n"
            "Example: " C_GREEN "imginf -/path/to/photo.jpg\n" C_RESET);
        return EXIT_FAILURE;
    }

    process_file(filepath, &opts);
    return EXIT_SUCCESS;
}