#include "main.cuh"

#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Not enough argument :\nUsage: " << argv[0] << " <cpu | kernel | indus>" << std::endl;
        return 1;
    }

    if (strcmp(argv[1], "cpu") == 0)
        return main_cpu();
    if (strcmp(argv[1], "kernel") == 0)
        return main_kernel();
    if (strcmp(argv[1], "indus") == 0)
        return main_indus();
    
    std::cerr << "Invalid argument :" << argv[1] << "\nUsage: " << argv[0] << " <cpu | kernel | indus>" << std::endl;
    return 1;
}
