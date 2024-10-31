#include "kernel/main_kernel.cuh"
#include "indus/main_indus.cuh"
#include "cpu/main_cpu.cuh"

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <cpu | kernel | indus>" << std::endl;
        return 1;
    }

    if (argv[1] == "cpu")
        return main_cpu();
    if (argv[1] == "kernel")
        return main_kernel();
    if (argv[1] == "indus")
        return main_indus();
    
    std::cerr << "Usage: " << argv[0] << " <cpu | kernel | indus>" << std::endl;
    return 1;
}
