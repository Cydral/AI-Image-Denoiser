#ifndef DNN_MODEL_H
#define DNN_MODEL_H

// Inclusion de bibliothèques nécessaires
#include <iostream>
#include <string>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/matrix.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/statistics.h>
#include <dlib/dir_nav.h>

using namespace std;
using namespace dlib;

using gray_pixel = uint8_t;
using rgb565_pixel = uint16_t;
const size_t std_image_size = 227;

template<typename T1, typename T2>
constexpr auto uint8_to_uint16(T1 high, T2  low) { return (((static_cast<uint16_t>(high)) << 8) | (static_cast<uint16_t>(low))); }

// Introduce the building blocks used to define the U-Net network
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using blockt = BN<dlib::cont<N, 3, 3, 1, 1, dlib::relu<BN<dlib::cont<N, 3, 3, stride, stride, SUBNET>>>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_up = dlib::add_prev2<dlib::cont<N, 2, 2, 2, 2, dlib::skip1<dlib::tag2<blockt<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, typename SUBNET> using res = dlib::relu<residual<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using res_down = dlib::relu<residual_down<block, N, dlib::bn_con, SUBNET>>;
template <int N, typename SUBNET> using res_up = dlib::relu<residual_up<block, N, dlib::bn_con, SUBNET>>;

// ----------------------------------------------------------------------------------------
template <typename SUBNET> using res64  = res<64, SUBNET>;
template <typename SUBNET> using res128 = res<128, SUBNET>;
template <typename SUBNET> using res256 = res<256, SUBNET>;
template <typename SUBNET> using res512 = res<512, SUBNET>;

template <typename SUBNET> using level1 = dlib::repeat<1, res64, res<64, SUBNET>>;
template <typename SUBNET> using level2 = dlib::repeat<1, res128, res_down<128, SUBNET>>;
template <typename SUBNET> using level3 = dlib::repeat<3, res256, res_down<256, SUBNET>>;
template <typename SUBNET> using level4 = dlib::repeat<2, res512, res_down<512, SUBNET>>;

template <typename SUBNET> using level1t = dlib::repeat<1, res64, res_up<64, SUBNET>>;
template <typename SUBNET> using level2t = dlib::repeat<1, res128, res_up<128, SUBNET>>;
template <typename SUBNET> using level3t = dlib::repeat<3, res256, res_up<256, SUBNET>>;
template <typename SUBNET> using level4t = dlib::repeat<2, res512, res_up<512, SUBNET>>;

// ----------------------------------------------------------------------------------------
template <
    template<typename> class TAGGED,
    template<typename> class PREV_RESIZED,
    typename SUBNET
>
using resize_and_concat = dlib::add_layer<
    dlib::concat_<TAGGED, PREV_RESIZED>,
    PREV_RESIZED<dlib::resize_prev_to_tagged<TAGGED, SUBNET>>>;

template <typename SUBNET> using utag1 = dlib::add_tag_layer<2100 + 1, SUBNET>;
template <typename SUBNET> using utag2 = dlib::add_tag_layer<2100 + 2, SUBNET>;
template <typename SUBNET> using utag3 = dlib::add_tag_layer<2100 + 3, SUBNET>;
template <typename SUBNET> using utag4 = dlib::add_tag_layer<2100 + 4, SUBNET>;

template <typename SUBNET> using utag1_ = dlib::add_tag_layer<2110 + 1, SUBNET>;
template <typename SUBNET> using utag2_ = dlib::add_tag_layer<2110 + 2, SUBNET>;
template <typename SUBNET> using utag3_ = dlib::add_tag_layer<2110 + 3, SUBNET>;
template <typename SUBNET> using utag4_ = dlib::add_tag_layer<2110 + 4, SUBNET>;

template <typename SUBNET> using concat_utag1 = resize_and_concat<utag1, utag1_, SUBNET>;
template <typename SUBNET> using concat_utag2 = resize_and_concat<utag2, utag2_, SUBNET>;
template <typename SUBNET> using concat_utag3 = resize_and_concat<utag3, utag3_, SUBNET>;
template <typename SUBNET> using concat_utag4 = resize_and_concat<utag4, utag4_, SUBNET>;

// ----------------------------------------------------------------------------------------
template <typename SUBNET> using generator_backbone =
relu<bn_con<cont<64, 7, 7, 2, 2,
    concat_utag1<level1t<
    concat_utag2<level2t<
    concat_utag3<level3t<
    concat_utag4<level4t<
    level4<utag4<
    level3<utag3<
    level2<utag2<
    level1<
    max_pool<3, 3, 2, 2, utag1<relu<bn_con<con<64, 7, 7, 2, 2, SUBNET>>>>>>>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------
// RGB to grayscale image conversion
void rgb_image_to_grayscale_image(const matrix<dlib::rgb_pixel>& rgb_image, matrix<gray_pixel>& gray_image) {
    gray_image.set_size(rgb_image.nr(), rgb_image.nc());
    std::transform(rgb_image.begin(), rgb_image.end(), gray_image.begin(),
        [](rgb_pixel a) {return gray_pixel(a.red * 0.299f + a.green * 0.587f + a.blue * 0.114f); });
}

// RGB image <=> RGB565 image
void rgb_image_to_rgb565_image(const matrix<rgb_pixel>& rgb_image, matrix<rgb565_pixel>& rgb565_image) {
    rgb565_image.set_size(rgb_image.nr(), rgb_image.nc());
    std::transform(rgb_image.begin(), rgb_image.end(), rgb565_image.begin(), [](const rgb_pixel& p) {
        return (static_cast<uint16_t>((p.red >> 3) << 11) | static_cast<uint16_t>((p.green >> 2) << 5) | static_cast<uint16_t>(p.blue >> 3));
    });
}
void rgb565_image_to_rgb_image(const matrix<rgb565_pixel>& rgb565_image, matrix<rgb_pixel>& rgb_image) {
    rgb_image.set_size(rgb565_image.nr(), rgb565_image.nc());
    std::transform(rgb565_image.begin(), rgb565_image.end(), rgb_image.begin(), [](const uint16_t& p) {
        uint8_t red = static_cast<uint8_t>(((p >> 11) & 0x1F) << 3);
        uint8_t green = static_cast<uint8_t>(((p >> 5) & 0x3F) << 2);
        uint8_t blue = static_cast<uint8_t>((p & 0x1F) << 3);
        return rgb_pixel(red, green, blue);
    });
}
void reduce_colors(matrix<rgb_pixel>& rgb_image) {
    matrix<rgb565_pixel> rgb565_image;
    rgb_image_to_rgb565_image(rgb_image, rgb565_image);
    rgb565_image_to_rgb_image(rgb565_image, rgb_image);
}

// Function to quantize a value to n bits (0 to 2^n-1) to a int_16
inline uint16_t quantize_n_bits(float value, int n) {
    // Ensure n is within a valid range
    if (n <= 0 || n > 16) throw std::invalid_argument("Invalid number of bits for quantization");    
    float max_value = (1 << n) - 1; // Calculate the maximum value for n bits    
    return static_cast<uint16_t>(std::round(value * max_value / 255.0f)); // Quantize the value
}
// Function to dequantize a value from n bits (0 to 2^n-1) to a float
inline float dequantize_n_bits(uint16_t quantized_value, int n) {
    // Ensure n is within a valid range
    if (n <= 0 || n > 16) throw std::invalid_argument("Invalid number of bits for dequantization");    
    const float max_value = (1 << n) - 1; // Calculate the maximum value for n bits    
    return (static_cast<float>(quantized_value) * 255.0f / max_value); // Dequantize the value
}

// Function to calculate the normalized weight for a pixel in Lab color space
float calc_weight(uint16_t quantized_a, uint16_t quantized_b, int n, float original_a, float original_b, float luminance) {
    // Calculate the Euclidean distance between the original and quantized values in Lab space
    static const float max_distance = std::sqrt(std::pow(255.0f, 2) + std::pow(255.0f, 2));
    float dequantized_a = dequantize_n_bits(quantized_a, n);
    float dequantized_b = dequantize_n_bits(quantized_b, n);
    float distance = std::sqrt(std::pow(original_a - dequantized_a, 2) + std::pow(original_b - dequantized_b, 2));    

    // Calculate the normalized weight as the ratio of the distance to the maximum distance, weighted by luminance
    if (luminance < 0.4f) luminance = 0.4f;
    float normalized_weight = (1.0f - (distance / max_distance)) * luminance + ((original_a / 255.0f) + (original_b / 255.0f)) / 2.0f;
    if (normalized_weight < 0.0f) normalized_weight = 0.0f;
    else if (normalized_weight > 1.0f) normalized_weight = 1.0f;
    return normalized_weight;
}

#endif // DNN_MODEL_H
