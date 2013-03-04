################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../build/classes/openmp.examples/block.c \
../build/classes/openmp.examples/firstprivate.c \
../build/classes/openmp.examples/howaboutthat.c \
../build/classes/openmp.examples/loop.c \
../build/classes/openmp.examples/noscope.c \
../build/classes/openmp.examples/private.c 

OBJS += \
./build/classes/openmp.examples/block.o \
./build/classes/openmp.examples/firstprivate.o \
./build/classes/openmp.examples/howaboutthat.o \
./build/classes/openmp.examples/loop.o \
./build/classes/openmp.examples/noscope.o \
./build/classes/openmp.examples/private.o 

C_DEPS += \
./build/classes/openmp.examples/block.d \
./build/classes/openmp.examples/firstprivate.d \
./build/classes/openmp.examples/howaboutthat.d \
./build/classes/openmp.examples/loop.d \
./build/classes/openmp.examples/noscope.d \
./build/classes/openmp.examples/private.d 


# Each subdirectory must supply rules for building sources it contributes
build/classes/openmp.examples/%.o: ../build/classes/openmp.examples/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


