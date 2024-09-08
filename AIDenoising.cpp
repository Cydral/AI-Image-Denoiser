/* AI-based Image Denoising Program using a Pix2Pix-like Algorithm with U-Net.
 * This program takes a noisy grayscale image as input and generates a denoised image as output.
 *
 * The main objective of this program is to utilize a Deep Neural Network (DNN) model based on the U-Net architecture.
 * U-Net is a popular model for image segmentation and image restoration tasks.
 *
 * In this context, it will be used to restore noisy images by removing noise while preserving essential details.
 * The Pix2Pix model is employed for learning the relationship between noisy input images and the desired denoised outputs.
 * The use of artificial intelligence (AI) in this program enhances image quality by eliminating noise artifacts,
 * which can be particularly beneficial in areas such as medical image restoration, digital photography, and various other applications.
 *
 * The input image is assumed to be grayscale, but the algorithm can be adapted to process color images by adjusting the model
 * and data preprocessing accordingly.
 */

#include "Model.h"

#include <algorithm>
#include <random>
#include <fstream>
#include <iterator>
#include <thread>
#include <filesystem>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

// ----------------------------------------------------------------------------------------
using net_backbone = generator_backbone<input<matrix<gray_pixel>>>;
using net_type = loss_multiclass_log_per_pixel<cont<256, 1, 1, 1, 1, net_backbone>>;

struct training_sample {
    matrix<rgb_pixel> source_image;
    matrix<gray_pixel> input_image;
    matrix<uint16_t> output_image;
};

// ----------------------------------------------------------------------------------------
rectangle make_random_cropping_rect(const matrix<rgb_pixel>& img, dlib::rand& rnd) {
    // figure out what rectangle we want to crop from the image
    double mins = 0.7, maxs = 1.0;
    auto scale = mins + rnd.get_random_double() * (maxs - mins);
    auto size = scale * std::min(img.nr(), img.nc());
    rectangle rect(size, size);
    // randomly shift the box around
    point offset(rnd.get_random_32bit_number() % (img.nc() - rect.width()), rnd.get_random_32bit_number() % (img.nr() - rect.height()));
    return move_rect(rect, offset);
}

// ----------------------------------------------------------------------------------------
void add_noise_to_image(const matrix<gray_pixel>& input_image, matrix<gray_pixel>& output_image, dlib::rand& rnd, double noise_prob, double blur_prob, double hole_prob) {
    bool do_transformation = false;
    if (rnd.get_double_in_range(0.0, 1.0) < blur_prob) {
        gaussian_blur(input_image, output_image, rnd.get_double_in_range(0.8, 1.1));
        do_transformation = true;
    } else assign_image(output_image, input_image);
    if (rnd.get_double_in_range(0.0, 1.0) < noise_prob) {
        for (long r = 0; r < output_image.nr(); ++r) {
            for (long c = 0; c < output_image.nc(); ++c) {
                const bool addition = (rnd.get_double_in_range(0.0, 1.0) > 0.5);
                if (rnd.get_double_in_range(0.0, 1.0) < noise_prob) {
                    if (addition) output_image(r, c) = ((uint16_t)output_image(r, c) + rnd.get_random_8bit_number()) / 2;
                    else output_image(r, c) = rnd.get_random_8bit_number();
                }
            }
        }
        do_transformation = true;
    }
    if (rnd.get_double_in_range(0.0, 1.0) < hole_prob) {
        const long nb_rect = rnd.get_integer_in_range(10, 27);
        for (long i = 0; i < nb_rect; ++i) {
            long rect_x = rnd.get_random_32bit_number() % output_image.nc();
            long rect_y = rnd.get_random_32bit_number() % output_image.nr();
            long rect_width = rnd.get_random_32bit_number() % 30;
            long rect_height = rnd.get_random_32bit_number() % 30;
            if (rect_x + rect_width >= output_image.nc()) rect_width = output_image.nc() - rect_x - 1;
            if (rect_y + rect_height >= output_image.nr()) rect_height = output_image.nr() - rect_y - 1;
            const bool addition = (rnd.get_double_in_range(0.0, 1.0) > 0.5);
            uint16_t p = rnd.get_integer_in_range(0, 255);
            for (long r = rect_y; r < (rect_y + rect_height); ++r) {
                for (long c = rect_x; c < (rect_x + rect_width); ++c) {
                    if (addition) output_image(r, c) = ((uint16_t)output_image(r, c) + p) / 2;
                    else output_image(r, c) = p;
                }
            }
        }
        do_transformation = true;
    }
    if (!do_transformation) gaussian_blur(input_image, output_image);
}
void randomly_crop_image(const matrix<rgb_pixel>& input_image, training_sample& crop, dlib::rand& rnd) {
    const auto rect = make_random_cropping_rect(input_image, rnd);
    const chip_details chip_details(rect, chip_dims(std_image_size, std_image_size));
    
    extract_image_chip(input_image, chip_details, crop.source_image, interpolate_bilinear());
    if (rnd.get_random_double() > 0.5) crop.source_image = fliplr(crop.source_image);

    matrix<gray_pixel> gray_image;
    rgb_image_to_grayscale_image(crop.source_image, gray_image);
    add_noise_to_image(gray_image, crop.input_image, rnd, 0.3, 0.5, 0.4);
    assign_image(crop.output_image, gray_image);
}

// ----------------------------------------------------------------------------------------
bool is_directory(const std::string& path) {
    struct stat s;
    if (stat(path.c_str(), &s) == 0) {
        if (s.st_mode & S_IFDIR) {
            return true;
        }
    }
    return false;
}
void parse_directory(const std::string& path, std::vector<std::string>& files) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) files.push_back(entry.path().string());
    }
}

// ----------------------------------------------------------------------------------------
template <typename pixel_type>
void scale_image(long nr, long nc, matrix<pixel_type>& dst) {
    matrix<pixel_type> resized(nr, nc);
    resize_image(dst, resized, interpolate_bilinear());
    dlib::assign_image(dst, resized);
}
template <typename pixel_type>
void resize_inplace(matrix<pixel_type>& inout, long size) {
    if (inout.nr() != size || inout.nc() != size) {
        matrix<pixel_type> mem_img;
        mem_img.set_size(size, size);
        resize_image(inout, mem_img);
        inout = mem_img;
    }
}
template <typename pixel_type>
void resize_max(matrix<pixel_type>& in, size_t max_image_dims) {
    size_t width = in.nc(), height = in.nr();
    if (width > max_image_dims || height > max_image_dims) {
        const double resize_factor = std::min(max_image_dims / (double)width, max_image_dims / (double)height);
        const size_t new_width = static_cast<size_t>(width * resize_factor);
        const size_t new_height = static_cast<size_t>(height * resize_factor);
        matrix<pixel_type> size_img(new_height, new_width);
        resize_image(in, size_img);
        in = size_img;
    }
}
bool is_two_small(const matrix<rgb_pixel>& image) {
    const size_t min_image_size = std_image_size;
    return (image.nc() < min_image_size || image.nr() < min_image_size);
}

// ----------------------------------------------------------------------------------------
std::atomic<bool> g_interrupted = false;
BOOL WINAPI CtrlHandler(DWORD ctrlType) {
    if (ctrlType == CTRL_C_EVENT) {
        g_interrupted = true;
        return TRUE;
    }
    return FALSE;
}

// ----------------------------------------------------------------------------------------
int main(int argc, char** argv) try {
    const long update_display = 25;
    const long max_minutes_elapsed = 10;
    double initial_learning_rate = 1e-1;
    size_t minibatch_size = 20, patience = 10000;

    po::options_description desc("Program options");
    desc.add_options()
        ("train", po::value<string>(), "train the denoising model <dir>")
        ("test", po::value<string>(), "test the denoising model <dir or file>")
        ("initial-learning-rate", po::value<double>(&initial_learning_rate)->default_value(1e-1), "set the initial learning rate (default 0.1)")
        ("minibatch-size", po::value<size_t>(&minibatch_size)->default_value(20), "set the minibatch size (default 20)")
        ("patience", po::value<size_t>(&patience)->default_value(10000), "set the patience parameter (default 10000)")
        ("help", "this help message");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << std::endl;
        return EXIT_FAILURE;
    }
    std::srand(std::time(nullptr));
    dlib::rand rnd(std::rand());
    size_t iteration = 0;
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
    set_dnn_prefer_smallest_algorithms();

    if (vm.count("train")) {
        const string input_dir = vm["train"].as<string>();
        const std::vector<file> training_images = dlib::get_files_in_directory_tree(input_dir, dlib::match_endings(".jpg .JPG .jpeg .JPEG"));
        if (training_images.size() == 0) {
            cout << "Didn't find images for the training dataset" << endl;
            return EXIT_FAILURE;
        }

        // Instantiate the model
        const string model_sync_filename = fs::current_path().string() + "/dnn_bw_denoising_filter.sync";   
        const string model_name = fs::current_path().string() + "/dnn_bw_denoising_filter.dnn";
        net_type my_net;
        if (file_exists(model_name)) deserialize(model_name) >> my_net;

        const double min_learning_rate = 1e-4;
        const double weight_decay = 1e-4;
        const double momentum = 0.9;

        // Initialize the trainer
        dnn_trainer<net_type> trainer(my_net, sgd(weight_decay, momentum));
        trainer.set_learning_rate(initial_learning_rate);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_mini_batch_size(minibatch_size);
        trainer.set_iterations_without_progress_threshold(patience);
        trainer.set_synchronization_file(model_sync_filename, std::chrono::minutes(max_minutes_elapsed));
        trainer.set_min_learning_rate(min_learning_rate);
        trainer.be_verbose();
        set_all_bn_running_stats_window_sizes(my_net, 1000);
        double cur_learning_rate = trainer.get_learning_rate();
        disable_duplicative_biases(my_net);

        // Output training parameters
        training_sample sample;
        matrix<rgb_pixel> input_image;
        try {
            const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
            load_image(input_image, image_info.full_name());
            randomly_crop_image(input_image, sample, rnd);
            my_net(sample.input_image);
            cout << my_net << std::endl;
            cout << "The network has " << my_net.num_layers << " layers in it" << std::endl;
            cout << std::endl << trainer << std::endl;
        } catch(...) {}        
        // Total images in the dataset
        cout << "images in dataset: " << training_images.size() << endl;

        // Use some threads to preload images
        dlib::pipe<training_sample> data(minibatch_size * 2);
        auto f = [&data, &training_images](time_t seed) {
            dlib::rand rnd(time(nullptr) + seed);
            matrix<rgb_pixel> input_image;
            training_sample temp;
            while (data.is_enabled()) {
                const auto& image_info = training_images[rnd.get_random_32bit_number() % training_images.size()];
                try {
                    load_image(input_image, image_info.full_name());
                    randomly_crop_image(input_image, temp, rnd);
                    data.enqueue(temp);
                } catch (...) {
                    cerr << "Error during image loading: " << image_info.full_name() << endl;
                }
            }
        };
        std::thread data_loader1([f]() { f(1); });
        std::thread data_loader2([f]() { f(2); });
        std::thread data_loader3([f]() { f(3); });
        cout << "Waiting for the initial pipe loading... ";
        while (data.size() < (minibatch_size * 2)) std::this_thread::sleep_for(std::chrono::seconds(1));
        cout << "done" << endl;
        
        std::vector<matrix<rgb_pixel>> sources;
        std::vector<matrix<gray_pixel>> samples;
        std::vector<matrix<uint16_t>> labels;
        dlib::image_window win;
        while (!g_interrupted && cur_learning_rate >= min_learning_rate) {
            // Train
            sources.clear();
            samples.clear();
            labels.clear();
            while (samples.size() < minibatch_size) {
                data.dequeue(sample);
                sources.push_back(sample.source_image);
                samples.push_back(sample.input_image);
                labels.push_back(sample.output_image);
            }
            if (++iteration % update_display == 0) { // We should see that the generated images start looking like samples
                std::vector<matrix<rgb_pixel>> disp_imgs;
                matrix<rgb_pixel> generated_image, noisy_image, source_image, image_to_save;
                matrix<gray_pixel> temp_image;
                size_t pos_i = 0, max_iter = __min(samples.size(), 4);
                trainer.get_net(dlib::force_flush_to_disk::no);
                auto gen_samples = my_net(samples);
                for (auto& image : gen_samples) {
                    rgb_image_to_grayscale_image(sources[pos_i], temp_image);
                    assign_image(source_image, temp_image);
                    assign_image(noisy_image, samples[pos_i]);
                    assign_image(generated_image, image);
                    disp_imgs.push_back(join_rows(source_image, join_rows(noisy_image, generated_image)));
                    if (++pos_i >= max_iter) break;
                }                
                image_to_save = tile_images(disp_imgs);
                save_jpeg(image_to_save, "dnn_bw_denoising_samples.jpg", 95);
                win.set_image(image_to_save);
                win.set_title("AI-DENOISING - Supervised-learning process, step#: " + to_string(iteration) + " - " + to_string(max_iter) + " samples - Original | Noisy | Denoised");
            }
            trainer.train_one_step(samples, labels);
            cur_learning_rate = trainer.get_learning_rate();
        }
        data.disable();
        data_loader1.join();
        data_loader2.join();
        data_loader3.join();
        trainer.get_net();
        my_net.clean();
        serialize(model_name) << my_net;
    } else if (vm.count("test")) {
        const string input_dir = vm["test"].as<string>();
        std::vector<string> images;
        if (!is_directory(input_dir)) images.push_back(input_dir);
        else parse_directory(input_dir, images);        
        cout << "total images to apply the denoising filter: " << images.size() << endl;
        if (images.size() == 0) {
            cout << "Didn't find images to colorify" << endl;
            return EXIT_FAILURE;
        }

        // Load the model
        const string model_name = fs::current_path().string() + "/dnn_bw_denoising_filter.dnn";
        net_type my_net;
        if (file_exists(model_name)) deserialize(model_name) >> my_net;
        else {
            cout << "Didn't find the precomputed model: " << model_name << endl;
            return EXIT_FAILURE;
        }       

        dlib::image_window win;
        matrix<rgb_pixel> input_image, generated_image, display_gray_image, image_to_save;
        matrix<gray_pixel> gray_image, temp_gray_image;
        for (auto& i : images) {
            try { load_image(input_image, i); }
            catch (...) {
                cerr << "Error during image loading: " << i << endl;
                continue;
            }
            if (is_two_small(input_image)) continue;
            resize_max(input_image, std_image_size * 2);
            rgb_image_to_grayscale_image(input_image, gray_image);
            assign_image(display_gray_image, gray_image);
            // --- Core process for donoising process
            {
                assign_image(temp_gray_image, gray_image);
                resize_inplace(temp_gray_image, std_image_size);
                matrix<uint16_t> output = my_net(temp_gray_image);
                assign_image(generated_image, output);
                scale_image(gray_image.nr(), gray_image.nc(), generated_image);
            }
            // ---
            image_to_save = join_rows(display_gray_image, generated_image);
            win.set_title("AI-DENOISING - Original (grayscale) " + to_string(display_gray_image.nc()) + "x" + to_string(display_gray_image.nr()) + ") | Generated (" + to_string(generated_image.nc()) + "x" + to_string(generated_image.nr()) + ")");
            win.set_image(image_to_save);
            save_jpeg(image_to_save, "dnn_bw_denoising_samples.jpg", 95);
            cout << i << " - Hit enter to process the next image or 'q' to quit";
            char c = std::cin.get();
            if (c == 'q' || c == 'Q') break;
        }
    }
} catch (std::exception& e) {
    cout << e.what() << endl;
}
