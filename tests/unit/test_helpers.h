#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void name(void)

#define ASSERT_INT_EQ(expected, actual) do { \
    int _e = (expected), _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL %s:%d: expected %d, got %d\n", \
                __FILE__, __LINE__, _e, _a); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_STR_EQ(expected, actual) do { \
    const char *_e = (expected), *_a = (actual); \
    if (strcmp(_e, _a) != 0) { \
        fprintf(stderr, "  FAIL %s:%d: expected \"%s\", got \"%s\"\n", \
                __FILE__, __LINE__, _e, _a); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL %s:%d: expected true\n", \
                __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NULL(ptr) do { \
    if ((ptr) != NULL) { \
        fprintf(stderr, "  FAIL %s:%d: expected NULL\n", \
                __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "  FAIL %s:%d: expected non-NULL\n", \
                __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define RUN_TEST(name) do { \
    int _prev_failed = tests_failed; \
    tests_run++; \
    printf("  RUN  %s\n", #name); \
    name(); \
    if (tests_failed == _prev_failed) { \
        tests_passed++; \
        printf("  PASS %s\n", #name); \
    } \
} while(0)

static inline int test_report(void) {
    printf("\n%d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(", %d FAILED", tests_failed);
    }
    printf("\n");
    return tests_failed > 0 ? 1 : 0;
}

#endif /* TEST_HELPERS_H */
