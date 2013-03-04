################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cuda.examples/smooth.cuda/smooth.cpp 

OBJS += \
./src/cuda.examples/smooth.cuda/smooth.o 

CPP_DEPS += \
./src/cuda.examples/smooth.cuda/smooth.d 


# Each subdirectory must supply rules for building sources it contributes
src/cuda.examples/smooth.cuda/%.o: ../src/cuda.examples/smooth.cuda/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


