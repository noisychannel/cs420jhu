################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../build/classes/cuda.examples/smooth.cuda/smooth.cpp 

OBJS += \
./build/classes/cuda.examples/smooth.cuda/smooth.o 

CPP_DEPS += \
./build/classes/cuda.examples/smooth.cuda/smooth.d 


# Each subdirectory must supply rules for building sources it contributes
build/classes/cuda.examples/smooth.cuda/%.o: ../build/classes/cuda.examples/smooth.cuda/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


