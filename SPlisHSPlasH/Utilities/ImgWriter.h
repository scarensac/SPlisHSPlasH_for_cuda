#ifndef IMG_WRITER
#define IMG_WRITER

#include <string>

void save_image(const ::std::string& name, float* img_vals, int w, int h, int channels_count, bool inver_y_axis);

void save_image(const ::std::string& name,  char* img_vals, int w, int h, int channels_count, bool inver_y_axis);

#endif