#include <chrono>
#include <iostream>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <vector>
#include <complex>
#include <cmath>
#include <mpi.h>
#include <cstring>

using namespace std;
MPI_Status status;
MPI_Request request;

int total_size;
int info[5];

int **alloc_2d_init(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

// int **local_result = alloc_2d_init(1000, 1000);
int local_result [1000*1000];
int imm_result[1000*1000];

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

void calculate(int *buffer, int size, int len, int scale, double x_center, double y_center, int k_value, int rank) {
    double cx = static_cast<double>(size) / 2 + x_center;
    double cy = static_cast<double>(size) / 2 + y_center;
    double zoom_factor = static_cast<double>(size) / 4 * scale;
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < size; ++j) {
            double x = (static_cast<double>(j) - cx) / zoom_factor;
            double y = (static_cast<double>(i) + (rank-1)*len - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < k_value);
            buffer[i*size + j] = k;
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
    MPI_Comm_size(MPI_COMM_WORLD, &total_size);
    int slave_size = total_size - 1;

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
                    // calculate(canvas, size, scale, center_x, center_y, k_value);
                    int local_length = ceil(double(size)/3);
                    int slave_rank;
                    info[0] = size;
                    info[1] = center_x;
                    info[2] = center_y;
                    info[3] = scale;
                    info[4] = k_value;

                    for (int m=0; m<slave_size; m++){
                        MPI_Isend(&info, 5, MPI_INT, (m+1), 0, MPI_COMM_WORLD, &request);
                    }

                    // int **imm_result;
                    // imm_result = alloc_2d_init(local_length, size);
                    // int imm_result[local_length * size];
                    for (int k=0; k<slave_size; k++){
                        MPI_Recv(&(imm_result[0]), local_length*size, MPI_INT,MPI_ANY_SOURCE, 1, MPI_COMM_WORLD,&status);
                        slave_rank = status.MPI_SOURCE;
                        int base_index = (slave_rank-1)*local_length;

                        for(int n=0; n<local_length; n++){
                            for(int q=0; q<size; q++){
                                canvas[{base_index+n,q}] = imm_result[n*size+q];
                            }
                        }  
                    }
                    // free(imm_result[0]);
                    // free(imm_result);                        

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
    else{
        // slave process.
        while(true){
            int re_info[5];
            MPI_Recv(&re_info, 5, MPI_INT,0,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int s_size, s_center_x, s_center_y, s_scale, s_k_value;
            s_size = re_info[0];
            s_center_x = re_info[1];
            s_center_y = re_info[2];
            s_scale = re_info[3];
            s_k_value = re_info[4];
            
            int local_size = ceil(double(s_size)/slave_size);
            // int local_result[local_size*s_size];

            calculate(local_result, s_size, local_size, s_scale,s_center_x,s_center_y,s_k_value, rank);
            MPI_Isend(&(local_result[0]), local_size*s_size, MPI_INT, 0,1, MPI_COMM_WORLD, &request);
        }
        
        // free(local_result[0]);
        // free(local_result);        
    }
    MPI_Finalize();
    return 0;
}
