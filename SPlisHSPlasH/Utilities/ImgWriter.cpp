#include "ImgWriter.h"

#include <fstream>
#include <iostream>

std::string as_pgm(const std::string& name) {
	if (!((name.length() >= 4)
		&& (name.substr(name.length() - 4, 4) == ".pgm")))
	{
		return name + ".pgm";
	}
	else {
		return name;
	}
}

void save_image(const ::std::string& name, float* img_vals, int w, int h, int channels_count, bool inver_y_axis)
{
	using ::std::ios;
	using ::std::ofstream;


	ofstream out(as_pgm(name), ios::binary | ios::out | ios::trunc);

	if (out.is_open()) {
		out << "P5\n" << w << " " << h << "\n255\n";
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				int sum = 0;
				for (int k = 0; k < channels_count; ++k) {
					int tmpval = static_cast<int>(::std::floor(256 * img_vals[((inver_y_axis ? (h - 1 - y) : y) * w + x) * channels_count + k]));
					if (tmpval < 0) { tmpval = 0u; }
					else if (tmpval > 255) { tmpval = 255; }
					sum += tmpval;
				}
				sum /= channels_count;
				const char outpv = static_cast<const char>(sum);
				out.write(&outpv, 1);
			}
		}
		out.close();
	}
	else {
		std::cout << "save_image:: failed opening the file  " << as_pgm(name) << std::endl;
		exit(-653);
		//throw(std::string("save_image:: failed opening the file  " + as_pgm(name)));
	}
}

void save_image(const ::std::string& name, char* img_vals, int w, int h, int channels_count, bool inver_y_axis)
{
	using ::std::ios;
	using ::std::ofstream;


	ofstream out(as_pgm(name), ios::binary | ios::out | ios::trunc);

	if (out.is_open()) {
		out << "P5\n" << w << " " << h << "\n255\n";
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				int sum = 0;
				for (int k = 0; k < channels_count; ++k) {
					sum += img_vals[((inver_y_axis ? (h - 1 - y) : y) * w + x) * channels_count + k];
				}
				sum /= channels_count;
				const char outpv = static_cast<const char>(sum);
				out.write(&outpv, 1);
			}
		}
		out.close();
	}
	else {
		throw(std::string("save_image:: failed opening the file  " + as_pgm(name)));
	}
}



