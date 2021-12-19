#include <chrono>
#include <iostream>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <vector>
#include <complex>
#include <mpi.h>
#include <cstring>

struct Square {
    std::vector<int> buffer;
    size_t length;

    explicit Square(size_t length) : buffer(length), length(length * length) {}

    void resize(size_t new_length) {
        buffer.assign(new_length * new_length, false);
        length = new_length;
    }

    auto& operator[](std::pair<size_t, size_t> pos) {
        return buffer[pos.second * length + pos.first];
    }
};

void calculate(Square &buffer, int size, int scale, double x_center, double y_center, int k_value) {
    double cx = static_cast<double>(size) / 2 + x_center;
    double cy = static_cast<double>(size) / 2 + y_center;
    double zoom_factor = static_cast<double>(size) / 4 * scale;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double x = (static_cast<double>(j) - cx) / zoom_factor;
            double y = (static_cast<double>(i) - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < k_value);
            buffer[{i, j}] = k;
        }
    }
}

static constexpr float MARGIN = 4.0f;
static constexpr float BASE_SPACING = 2000.0f;
static constexpr size_t SHOW_THRESHOLD = 500000000ULL;

int main(int argc, char **argv) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (0 == rank) {
        graphic::GraphicContext context{"Assignment 2"};
        Square canvas(100);
        size_t duration = 0;
        size_t pixels = 0;
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
            {
                auto io = ImGui::GetIO();
                ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
                ImGui::SetNextWindowSize(io.DisplaySize);
                ImGui::Begin("Assignment 2", nullptr,
                             ImGuiWindowFlags_NoMove
                             | ImGuiWindowFlags_NoCollapse
                             | ImGuiWindowFlags_NoTitleBar
                             | ImGuiWindowFlags_NoResize);
                ImDrawList *draw_list = ImGui::GetWindowDrawList();
                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                            ImGui::GetIO().Framerate);
                static int center_x = 0;
                static int center_y = 0;
                static int size = 800;
                static int scale = 1;
                static ImVec4 col = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
                static int k_value = 100;
                ImGui::DragInt("Center X", &center_x, 1, -4 * size, 4 * size, "%d");
                ImGui::DragInt("Center Y", &center_y, 1, -4 * size, 4 * size, "%d");
                ImGui::DragInt("Fineness", &size, 10, 100, 1000, "%d");
                ImGui::DragInt("Scale", &scale, 1, 1, 100, "%.01f");
                ImGui::DragInt("K", &k_value, 1, 100, 1000, "%d");
                ImGui::ColorEdit4("Color", &col.x);
                {
                    using namespace std::chrono;
                    auto spacing = BASE_SPACING / static_cast<float>(size);
                    auto radius = spacing / 2;
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    const ImU32 col32 = ImColor(col);
                    float x = p.x + MARGIN, y = p.y + MARGIN;
                    canvas.resize(size);
                    auto begin = high_resolution_clock::now();
                    calculate(canvas, size, scale, center_x, center_y, k_value);
                    auto end = high_resolution_clock::now();
                    pixels += size;
                    duration += duration_cast<nanoseconds>(end - begin).count();
                    if (duration > SHOW_THRESHOLD) {
                        std::cout << pixels << " pixels in last " << duration << " nanoseconds\n";
                        auto speed = static_cast<double>(pixels) / static_cast<double>(duration) * 1e9;
                        std::cout << "speed: " << speed << " pixels per second" << std::endl;
                        pixels = 0;
                        duration = 0;
                    }
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            if (canvas[{i, j}] == k_value) {
                                draw_list->AddCircleFilled(ImVec2(x, y), radius, col32);
                            }
                            x += spacing;
                        }
                        y += spacing;
                        x = p.x + MARGIN;
                    }
                }
                ImGui::End();
            }
        });
    }
    MPI_Finalize();
    return 0;
}
