#include "common.h"

int readFileToArray(const char *filename, char* data, int lineCount) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return -1;
    }

    // Read each line and store it in the single array
    for (int i = 0; i < lineCount; i++) {
        if (fgets(data + i * LINE_LENGTH, LINE_LENGTH, file) == NULL) {
            break; // Stop reading if EOF is reached unexpectedly
        }
    }

    fclose(file);

    return 0;
}

int writeFileFromArray(const char *filename, char* data, int lineCount) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        return -1;
    }

    for (int i = 0; i < lineCount; i++) {
        fprintf(file, "%c\n", *(data + i * LINE_LENGTH));
    }

    fclose(file);

    return 0;
}