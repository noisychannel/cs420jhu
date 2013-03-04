################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/openmp.examples/block.c \
../src/openmp.examples/firstprivate.c \
../src/openmp.examples/howaboutthat.c \
../src/openmp.examples/loop.c \
../src/openmp.examples/noscope.c \
../src/openmp.examples/private.c 

OBJS += \
./src/openmp.examples/block.o \
./src/openmp.examples/firstprivate.o \
./src/openmp.examples/howaboutthat.o \
./src/openmp.examples/loop.o \
./src/openmp.examples/noscope.o \
./src/openmp.examples/private.o 

C_DEPS += \
./src/openmp.examples/block.d \
./src/openmp.examples/firstprivate.d \
./src/openmp.examples/howaboutthat.d \
./src/openmp.examples/loop.d \
./src/openmp.examples/noscope.d \
./src/openmp.examples/private.d 


# Each subdirectory must supply rules for building sources it contributes
src/openmp.examples/%.o: ../src/openmp.examples/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


